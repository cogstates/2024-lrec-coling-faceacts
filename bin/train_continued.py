#!/usr/bin/env python
# -*- coding: utf-8 -*-
# mypy: ignore-errors
"""TODO"""
import collections
import contextlib
import functools
import hashlib
import itertools
import json
import logging
import os
import warnings
from glob import glob
from typing import Any, Iterator

import tokenizers
import torch
import pandas as pd
import lightning.pytorch as pl
from retry import retry
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import get_linear_schedule_with_warmup
from sklearn.exceptions import UndefinedMetricWarning

from eval import score
from src.core.app import harness
from src.core.context import Context
from src.core.path import dirparent
from src.core import nvidia
from src.data import cmu


@contextlib.contextmanager
def evaluating(model: torch.nn.Module) -> Iterator[None]:
    with torch.no_grad():
        try:
            yield model.eval()
        finally:
            model.train()


def get_predictions(model: pl.LightningModule, data_type: str) -> pd.DataFrame:
    model.model.to(model.device)
    data = model.get_dataloader(data_type)

    assert data.dataset.data_path.endswith(".csv")
    recs = pd.read_csv(data.dataset.data_path).to_dict("records")
    with evaluating(model.model):
        refs, preds = [], []
        desc = f"Predicting {data_type.capitalize()} (Epoch {model.current_epoch})"
        for batch in tqdm(data, desc=desc, leave=False):
            tgts = batch["target_ids"]
            outs = model.model.generate(
                input_ids=batch["source_ids"].to(model.device),
                attention_mask=batch["source_mask"].to(model.device),
                max_length=model.hparams["max_seq_length"],
            )
            refs.extend(model.tokenizer.batch_decode(tgts, skip_special_tokens=True))
            preds.extend(model.tokenizer.batch_decode(outs, skip_special_tokens=True))
    assert len(refs) == len(preds) and len(preds) == len(recs)
    ret = pd.DataFrame(
        [rec | {"ref": ref, "pred": pred} for rec, ref, pred in zip(recs, refs, preds)]
    )
    assert (ret["target_text"] == ret["ref"]).all() == True
    return ret


def evaluate_model(model: pl.LightningModule, data_type: str) -> pd.DataFrame:
    log = logging.getLogger(__name__)
    preds = get_predictions(model, data_type)
    path = os.path.join(
        model.hparams.outdir, f"epoch{model.current_epoch}", f"preds.{data_type}.csv"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    preds.to_csv(path, index=False)
    if data_type == "dev":
        model.log("dev_micro_f1", score(path)["f1"]["micro"])
        model.log("dev_macro_f1", score(path)["f1"]["macro"])
    log.info("wrote: %s", path)


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.hparams.update(hparams)

        model_name = self.hparams.model_name_or_path
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = get_tokenizer(model_name)

        self.validation_step_outputs = []
        self.training_step_outputs = []

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        lm_labels=None,
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch["target_mask"],
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.training_step_outputs.append(loss)
        return {
            "loss": loss,
        }

    def on_training_epoch_end(self):
        avg_train_loss = torch.stack(self.training_step_outputs).mean()
        self.training_step_outputs.clear()  # free memory
        return {
            "avg_train_loss": avg_train_loss,
        }

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.validation_step_outputs.append(loss)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.validation_step_outputs).mean()
        self.log("val_loss", avg_val_loss)
        self.validation_step_outputs.clear()  # free memory
        # TODO: This is inefficient since outputs are computed in _step... refactor.
        if self.global_step > 0:
            evaluate_model(self, "test")
            evaluate_model(self, "dev")
        return {
            "val_loss": avg_val_loss,
        }

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        self.opt = optimizer
        return [optimizer]

    def train_dataloader(self):
        train_dataset = get_dataset(
            tokenizer=self.tokenizer,
            data_dir=self.hparams.datadir,
            data_type="train",
            max_length=self.hparams["max_seq_length"],
        )
        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.hparams.train_batch_size,
            num_workers=4,
        )
        t_total = (
            (
                len(dataloader.dataset)
                // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu))
            )
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=t_total,
        )
        self.lr_scheduler = scheduler
        return dataloader

    def get_dataloader(self, data_type: str):
        val_dataset = get_dataset(
            tokenizer=self.tokenizer,
            data_dir=self.hparams.datadir,
            data_type=data_type,
            max_length=self.hparams["max_seq_length"],
        )
        return torch.utils.data.DataLoader(
            val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4
        )

    def val_dataloader(self):
        return self.get_dataloader("test")


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, tokenizer: T5Tokenizer, data_dir: str, data_type: str, max_length: int
    ):
        self.data_type = data_type
        self.data_path = os.path.join(data_dir, f"{data_type}.csv")
        self.max_length = max_length
        self.tokenizer = tokenizer

        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index][
            "attention_mask"
        ].squeeze()  # might need to squeeze
        target_mask = self.targets[index][
            "attention_mask"
        ].squeeze()  # might need to squeeze

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
        }

    def _build_examples(self):
        df = pd.read_csv(self.data_path)
        #  if self.data_type in ("dev", "test"):
        #      df = df[df.target_text.isin(cmu.FACEACTS)]  # XXX: Only return face act data for testing.
        inputs, targets = df["input_text"].to_list(), df["target_text"].to_list()

        for idx in range(len(inputs)):
            inp = inputs[idx]
            tgt = targets[idx]

            tokenized_input = self.tokenizer.batch_encode_plus(
                [inp],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
                [tgt],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)


def get_dataset(tokenizer: T5Tokenizer, data_dir: str, data_type: str, max_length: int):
    return Dataset(
        tokenizer=tokenizer,
        data_dir=data_dir,
        data_type=data_type,
        max_length=max_length,
    )


def get_tokenizer(model_name_or_path: str) -> T5Tokenizer:
    tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_tokens(tokenizers.AddedToken("\n", normalized=False))
    return tokenizer


@functools.cache
def get_max_seq_length(model_name_or_path: str, data_dir: str) -> int:
    tokenizer = get_tokenizer(model_name_or_path)
    max_seq_lengths = []
    for path in glob(os.path.join(data_dir, "*.csv")):
        data_type = os.path.basename(path).replace(".csv", "")
        data = pd.read_csv(path).input_text.to_list()

        input_ids = tokenizer.batch_encode_plus(
            data, padding="longest", truncation="do_not_truncate", return_tensors="pt"
        )["input_ids"]
        max_seq_lengths.append(input_ids.size(dim=1))
    if max(max_seq_lengths) > 256:
        list(map(print, ["WARNING: CLIPPING MAX SEQ LENGTH"] * 5))
        return 256
    return max(max_seq_lengths)


HPARAMS = collections.OrderedDict(
    {
        "seed": [
            #  21, 7,
            42,
        ],
        "batch_size": [
            #  1, 2, 4, 8, 16,
            #  32,
            16
        ],
        "gradient_accumulation_steps": [
            #  1, 2, 4, 8, 16, 32
            2,
        ],
        "learning_rate": [3e-4],
        "weight_decay": [0.01],
        "adam_epsilon": [1e-8],
        "model_name_or_path": [
            #  "/home/asoubki/dev/faceacts/output/sweep-all-reformat/8762876b45e5e1a30c932d22e82713af/last.ckpt",
            "/home/asoubki/dev/faceacts/output/sweep-all-reformat-thanking/2347b5c7c48e5635fda77b91d035076d/last.ckpt",
            #  "/home/asoubki/dev/faceacts/output/sweep-all-data/c86867b8a3323bfbcb4e09ba27a39b56/last.ckpt",
            #  "/home/asoubki/dev/faceacts/output/sweep-all-data-thanking/f1161dbc52848dc6a4ae106f37c44311/last.ckpt",
            #  "google/flan-t5-small",
            #  "google/flan-t5-base",
            #  "google/flan-t5-large",
            #  "t5-small",
            #  "t5-base",
            #  "t5-large",
            #  "google/flan-ul2",
        ],
        "warmup_steps": [0],
        "max_seq_length": [256],
        "gradient_clip_val": [
            #  None, 0.5,
            1.0,
        ],
        #  "datadir": []
    }
)


@retry(torch.cuda.OutOfMemoryError)
def run(hparams: dict[str, Any], num_train_epochs: int, outdir: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(nvidia.best_gpu())  # TODO: Remove.

    logger = logging.getLogger(__name__)
    logger.info("[HYPERPARAMETERS]\n%s", json.dumps(hparams, indent=2))
    pl.seed_everything(hparams["seed"])
    rid = hashlib.md5(str(sorted(hparams.items())).encode("utf-8")).hexdigest()
    outdir = os.path.join(outdir, rid)
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame([hparams]).to_csv(os.path.join(outdir, "hparams.csv"), index=False)
    logger.info("[HASH: %s]", rid)

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="dev_micro_f1",
        #  monitor="val_loss",
        min_delta=0.0,
        patience=3,
        mode="max",
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=outdir,
        filename="epoch{epoch}/model",
        auto_insert_metric_name=False,
        save_last=True,
        save_top_k=1,
    )

    trainer = pl.Trainer(
        accumulate_grad_batches=hparams["gradient_accumulation_steps"],
        accelerator="auto",
        #  devices=1,
        #  precision="bf16-mixed",
        min_epochs=10,
        max_epochs=num_train_epochs,
        gradient_clip_val=hparams["gradient_clip_val"],
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=False,
        enable_progress_bar=True,
        fast_dev_run=False,
    )
    #  model = T5FineTuner(
    #      hparams
    #      | {
    #          "n_gpu": int(os.environ["CUDA_VISIBLE_DEVICES"]),
    #          "outdir": outdir,
    #          "datadir": hparams["datadir"],
    #          "num_train_epochs": num_train_epochs,
    #          "train_batch_size": hparams["batch_size"],
    #          "eval_batch_size": max(hparams["batch_size"], 32),
    #      }
    #  )
    model_name_or_path = hparams["model_name_or_path"]
    hparams_path = os.path.join(os.path.dirname(model_name_or_path), "hparams.csv")
    hparams = pd.read_csv(hparams_path).to_dict("records")[0] | {
        "n_gpu": int(os.environ["CUDA_VISIBLE_DEVICES"]),
        "outdir": outdir,
        "datadir": hparams["datadir"],
        "num_train_epochs": num_train_epochs,
        "train_batch_size": hparams["batch_size"],
        "eval_batch_size": max(hparams["batch_size"], 32),
        "model_name_or_path_cont": model_name_or_path,
    }
    model = T5FineTuner.load_from_checkpoint(model_name_or_path, hparams=hparams)

    trainer.fit(model, ckpt_path="last")


def main(ctx: Context) -> None:
    default_outdir = os.path.join(dirparent(os.path.realpath(__file__), 2), "output")
    ctx.parser.add_argument(
        "datadirs", type=os.path.realpath, help="path to input data dir", nargs="+"
    )
    ctx.parser.add_argument("--num-train-epochs", type=int, default=30)
    ctx.parser.add_argument(
        "-o", "--outdir", type=os.path.realpath, default=default_outdir
    )
    args = ctx.parser.parse_args()

    warnings.simplefilter("ignore", UndefinedMetricWarning)
    torch.set_float32_matmul_precision("high")
    cfg = HPARAMS | {"datadir": args.datadirs}
    for prd in itertools.product(*cfg.values()):
        hparams = dict(zip(cfg.keys(), prd))
        #  hparams["max_seq_length"] = hparams.get(
        #      "max_seq_length", None
        #  ) or get_max_seq_length(hparams["model_name_or_path"], hparams["datadir"])
        run(
            hparams=hparams,
            num_train_epochs=args.num_train_epochs,
            outdir=args.outdir,
        )


if __name__ == "__main__":
    harness(main)
