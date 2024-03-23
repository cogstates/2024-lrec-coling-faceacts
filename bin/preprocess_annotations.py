#!/usr/bin/env python
# -*- coding: utf-8 -*-
# mypy: ignore-errors
"""An example script"""
import os
from glob import glob
from typing import Any

import numpy as np
import pandas as pd
from more_itertools import divide, flatten

from src.core.app import harness
from src.core.context import Context, get_context


def set_target_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"face_act": "face_act_adil"})

    def fn(row: pd.Series) -> pd.Series:
        # No double entries.
        assert (pd.isna(row.face_act_adil) and not pd.isna(row.face_act_shyne)) or (
            not pd.isna(row.face_act_adil) and pd.isna(row.face_act_shyne)
        ), row
        # get the annotation.
        face_act = (
            row.face_act_adil if not pd.isna(row.face_act_adil) else row.face_act_shyne
        )
        assert not pd.isna(face_act)
        row["face_act"] = face_act
        row["target_text"] = face_act
        return row

    return df.apply(fn, axis=1)


def load_datadir(dirpath: str) -> pd.DataFrame:
    ret = []
    for path in glob(os.path.join(dirpath, "*")):
        ret.append(pd.read_csv(path))
    return set_target_text(pd.concat(ret))


def cross_validation_split(data: pd.DataFrame, seed: int, n: int) -> pd.DataFrame:
    ret = []
    # Shuffle the conversation ids.
    cids = sorted(data.conversation_id.unique())
    np.random.seed(seed)
    np.random.shuffle(cids)
    # Split conversation ids into folds.
    folds = list(map(list, divide(n, cids)))
    assert len(folds) == n
    # Create folds.
    for fdx in range(len(folds)):
        train = list(flatten([folds[idx] for idx in range(len(folds)) if idx != fdx]))
        test = folds[fdx]
        assert len(cids) == len(train) + len(test)
        ret.append(
            pd.concat(
                [
                    data[data.conversation_id.isin(train)].assign(split="train"),
                    data[data.conversation_id.isin(test)].assign(split="test"),
                ]
            ).assign(fold=fdx)
        )
    return pd.concat(ret)


# TODO: Support stratified.
def save_cross_validation_splits(outdir: str, data: pd.DataFrame, seed: int, n: int):
    ctx = get_context()
    data = cross_validation_split(data, seed=seed, n=n)
    for fold in sorted(data.fold.unique()):
        df = data[data.fold == fold]
        # Print summary stats.
        summary = df.groupby(["split", "face_act"]).count()
        summary = summary.speaker.unstack(1).fillna(0).astype(int)
        summary.index.name, summary.columns.name = None, None
        summary = summary.assign(total=summary.T.sum())
        summary.loc["total"] = summary.sum()
        ctx.log.info(("-" * 29) + f" [Fold {fold}] " + ("-" * 29))
        list(map(ctx.log.info, str(summary).split("\n")))
        ctx.log.info("-" * 68)
        # Write to output directory.
        fold_outdir = os.path.join(
            outdir,
            f"seed{seed}.xval",
            f"fold-{fold}",
        )
        os.makedirs(fold_outdir, exist_ok=True)
        for split in ("train", "dev", "test"):
            outpath = os.path.join(fold_outdir, f"{split}.csv")
            split = "train" if split == "dev" else split
            df[df.split == split].to_csv(outpath, index=False)
            ctx.log.info(f"wrote: {outpath}")


def main(ctx: Context) -> None:
    ctx.parser.add_argument("datadir", type=os.path.realpath)
    ctx.parser.add_argument("-o", "--outdir", type=os.path.realpath, required=True)
    args = ctx.parser.parse_args()

    anns = load_datadir(args.datadir)
    save_cross_validation_splits(args.outdir, anns, seed=42, n=5)


if __name__ == "__main__":
    harness(main)
