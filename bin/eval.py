#!/usr/bin/env python
# -*- coding: utf-8 -*-
# mypy: ignore-errors
"""An example script"""
import collections
import os
import warnings
from glob import glob
from typing import Any

import evaluate
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm

from dgen import repair_malformed_label
from src.core.app import harness
from src.core.context import Context
from src.core.path import dirparent
from src.data import cmu


def compute_one(preds, refs, metric, labels):
    mtr = evaluate.load(metric)
    rslts = mtr.compute(predictions=preds, references=refs, average=None)[metric]
    macro = mtr.compute(predictions=preds, references=refs, average="macro")[metric]
    micro = mtr.compute(predictions=preds, references=refs, average="micro")[metric]
    return collections.OrderedDict(
        **{"macro": macro, "micro": micro},
        **{labels[idx]: rslt for idx, rslt in enumerate(rslts)},
    )


def compute(preds, refs, metrics, labels):
    assert len(preds) == len(refs)
    return {mtr: compute_one(preds, refs, mtr, labels) for mtr in metrics}


def score(path: str) -> dict[str, Any]:
    df = pd.read_csv(path, keep_default_na=False)
    df = df[df.ref.isin(cmu.FACEACTS)]  # XXX: Breaking for MTL analysis. Mark corpus.
    df = df.assign(pred=df.pred.apply(repair_malformed_label))

    metrics = ["f1", "precision", "recall"]
    labels = sorted(list(set(df.pred) | set(df.ref)))
    preds = list(map(lambda lbl: labels.index(lbl), df.pred))
    refs = list(map(lambda lbl: labels.index(lbl), df.ref))
    return compute(preds, refs, metrics, labels)


def main(ctx: Context) -> None:
    default_outdir = os.path.join(
        dirparent(os.path.realpath(__file__), 2), "data", "proc", "eval"
    )
    ctx.parser.add_argument("datadir", type=os.path.realpath)
    ctx.parser.add_argument(
        "-o", "--outdir", type=os.path.realpath, default=default_outdir
    )
    args = ctx.parser.parse_args()

    warnings.simplefilter("ignore", UndefinedMetricWarning)
    ret = []
    for hpath in tqdm(glob(os.path.join(args.datadir, "*", "hparams.csv"))):
        hparams = pd.read_csv(hpath).to_dict("records")
        assert len(hparams) == 1
        hparams = hparams[0]
        dirpath = os.path.dirname(hpath)
        for epath in glob(os.path.join(dirpath, "epoch*", "*.csv")):
            epoch = int(os.path.basename(os.path.dirname(epath)).replace("epoch", ""))
            split = os.path.basename(epath).split(".")[1]
            assert split in ("dev", "test")
            hparams |= {
                "epoch": epoch,
                "split": split,
                "data_dir": hparams["datadir"],  # XXX
                "preds_path": epath,
                "hparams_path": hpath,
                "checksum": os.path.basename(dirpath),
            }
            if os.path.basename(hparams["datadir"]).startswith("fold"):
                fold = int(os.path.basename(hparams["datadir"]).split("-")[1])
                dinfo = os.path.basename(dirparent(hparams["datadir"], 1)).split(".")
            else:
                fold = None
                dinfo = os.path.basename(hparams["datadir"]).split(".")
            assert dinfo[0].startswith("seed")
            # assert dinfo[1].startswith("hlen")  # XXX
            hparams |= {
                "split_seed": int(dinfo[0].replace("seed", "")),
                # "hlen": int(dinfo[1].replace("hlen", "")),  # XXX
                "data_version": "default" if len(dinfo) < 3 else dinfo[2],
                "fold": fold,
            }
            for metric, dct in score(epath).items():
                ret.append(hparams | {"metric": metric} | dct)
    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, "latest.csv")
    pd.DataFrame(ret).to_csv(outpath, index=False)
    ctx.log.info("wrote: %s", outpath)


if __name__ == "__main__":
    harness(main)
