#!/usr/bin/env python
# -*- coding: utf-8 -*-
# mypy: ignore-errors
"""An example script"""
import os
from typing import Any

import pandas as pd
import lightning.pytorch as pl
from more_itertools import chunked_even

from src.core.app import harness
from src.core.context import Context
from src.core.path import dirparent
from train import evaluating, T5FineTuner


def get_predictions(model: pl.LightningModule, data_path: str) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    recs = data.to_dict("records")
    with evaluating(model.model):
        preds = []
        for chunk in chunked_even(data["input_text"], 32):
            batch = model.tokenizer.batch_encode_plus(
                chunk,
                max_length=model.hparams["max_seq_length"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            outs = model.model.generate(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                max_length=model.hparams["max_seq_length"],
            )
            preds.extend(model.tokenizer.batch_decode(outs, skip_special_tokens=True))
    assert len(data) == len(preds)
    ret = pd.DataFrame(
        [rec | {"pred": pred} for rec, pred in zip(recs, preds)]
    )
    return ret


def main(ctx: Context) -> None:
    default_outdir = os.path.join(
        dirparent(os.path.realpath(__file__), 2), "data", "proc", "predict"
    )
    ctx.parser.add_argument("data_path", type=os.path.realpath)
    ctx.parser.add_argument("-m", "--model_path", type=os.path.realpath)
    ctx.parser.add_argument(
        "-o", "--outdir", type=os.path.realpath, default=default_outdir
    )
    args = ctx.parser.parse_args()

    hparams_path = os.path.join(os.path.dirname(args.model_path), "hparams.csv")
    hparams = pd.read_csv(hparams_path).to_dict("records")[0]
    model = T5FineTuner.load_from_checkpoint(args.model_path, hparams=hparams)
    preds = get_predictions(model, args.data_path)

    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, os.path.basename(args.data_path))
    preds.to_csv(outpath, index=False)
    ctx.log.info("wrote: %s", outpath)


if __name__ == "__main__":
    harness(main)
