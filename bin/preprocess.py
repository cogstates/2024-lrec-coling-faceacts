#!/usr/bin/env python
# -*- coding: utf-8 -*-
# mypy: ignore-errors
"""Preprocess CMU data for training"""
import os
from argparse import Namespace
from operator import itemgetter
from typing import Any, Iterable

import pandas as pd
from more_itertools import windowed

from src.core.app import harness
from src.core.context import Context
from src.core.path import dirparent
from src.data import cmu


def add_inline_columns(row: pd.Series) -> pd.Series:
    speaker = "EE" if row.speaker else "ER"
    row["input_text"] = f"{speaker}: {row.utterance}"
    row["target_text"] = f"{row.input_text} ({row.true_face})"
    return row


def preprocess_for_inline(data: pd.DataFrame, hlen: int) -> pd.DataFrame:
    ret = []
    data = data.apply(add_inline_columns, axis=1)
    for cid, win, wdx in conversation_windows(data, hlen):
        ret.append(
            {
                "input_text": "\n".join(map(itemgetter("input_text"), win)),
                "target_text": "\n".join(map(itemgetter("target_text"), win)),
                "conversation_id": cid,
                "index": data.conversation_id.unique().tolist().index(cid),
                "windex": wdx,
                "split": win[-1]["split"],
            }
        )
    ret = pd.DataFrame(ret).sort_values(["index", "windex"]).set_index("index")
    ret = ret.reset_index().drop(columns="index")
    return ret


def add_speaker_prefix(row: pd.Series) -> pd.Series:
    #  speaker = "EE" if row.speaker else "ER"
    if row.speaker in (1, "1"):
        speaker = "EE"
    elif row.speaker in (0, "0"):
        speaker = "ER"
    else:
        speaker = row.speaker
    #  elif row.speaker == "A":
    #      speaker = "A"
    #  elif row.speaker == "B":
    #      speaker = "B"
    #  else:
    #      raise ValueError
    row.utterance = f"{speaker}: {row.utterance}"
    return row


def add_task_prefix(row: pd.Series) -> pd.Series:
    prefix = "face acts" if row.target_text in cmu.FACEACTS else "dialog acts"
    prefix = "politeness" if row.corpus.endswith("politeness") else prefix
    row.input_text = f"{prefix}:\n{row.input_text}"
    return row


def add_turn_index(row: pd.Series) -> pd.Series:
    turn_idx = row["index"]  # int(row.turn_id.split("_")[1])
    row.utterance = f"{row.utterance} ({turn_idx})"
    return row


def preprocess_for_extraction(data: pd.DataFrame, hlen: int) -> pd.DataFrame:
    ret = []
    data = data.apply(add_speaker_prefix, axis=1)
    for cid, win, wdx in conversation_windows(data, hlen):
        ret.append(
            {
                "input_text": "\n".join(map(itemgetter("utterance"), win)),
                "target_text": win[-1]["true_face"],
                "conversation_id": cid,
                "index": data.conversation_id.unique().tolist().index(cid),
                "windex": wdx,
                "split": win[-1]["split"],
                "corpus": win[-1]["corpus"],
            }
        )
    ret = pd.DataFrame(ret).sort_values(["index", "windex"]).set_index("index")
    ret = ret.reset_index().drop(columns="index")
    return ret


def conversation_windows(
    data: pd.DataFrame, hlen: int
) -> Iterable[tuple[int, list[dict[str, Any]], int]]:
    padding = [None] * (hlen - 1)
    for cid, df in data.groupby("conversation_id"):
        for wdx, win in enumerate(
            windowed(padding + df.to_dict("records"), n=hlen, step=1)
        ):
            win = [w for w in win if w]
            assert len(set(map(itemgetter("split"), win))) == 1
            assert len(set(map(itemgetter("corpus"), win))) == 1
            yield cid, win, wdx


# XXX: This duplicates a lot of code...
def process_cross_validation(ctx: Context, args: Namespace) -> None:
    assert args.version.uses_cross_validation()
    data = cmu.load(args.seed, version=args.version)
    if not args.version.uses_mtl():
        data = data.assign(corpus="cmu")
    for fold in sorted(data.fold.unique()):
        df = data[data.fold == fold]
        # Print summary stats.
        summary = df.groupby(["split", "true_face"]).count()
        summary = summary.speaker.unstack(1).fillna(0).astype(int)
        summary.index.name, summary.columns.name = None, None
        summary = summary.assign(total=summary.T.sum())
        summary.loc["total"] = summary.sum()
        ctx.log.info(("-" * 29) + " [Fold %d] " + ("-" * 29), fold)
        list(map(ctx.log.info, str(summary.T).split("\n")))
        ctx.log.info("-" * 68)
        # Process to target format.
        #  df = df.apply(add_turn_index, axis=1)  # XXX: EXPERIMENTAL
        fmap = {
            "extraction": preprocess_for_extraction,
            "inline": preprocess_for_inline,
        }
        # XXX: Don't use history length for politeness corpus. (there is no context)
        if args.version.name.endswith("POL"):  # XXX: EXPERIMENTAL
            df = pd.concat([
                fmap[args.format](df[df.corpus != "cmu"], hlen=1),
                fmap[args.format](df[df.corpus == "cmu"], args.hlen)
            ]).reset_index(drop=True)
        else:
            df = fmap[args.format](df, args.hlen)
        if args.version.uses_mtl():
            df = df.apply(add_task_prefix, axis=1)  # XXX: ONLY FOR MTL
        version = "xval" if not args.version.uses_mtl() else "mtl"
        if args.version != cmu.Version.XVAL:
            version = f"{version}-with-{args.version.name.lower().split('_')[1]}"
        outdir = os.path.join(
            args.outdir,
            args.format,
            f"seed{args.seed}.hlen{args.hlen}.{version}",
            f"fold-{fold}",
        )
        os.makedirs(outdir, exist_ok=True)
        for split in ("train", "dev", "test"):
            outpath = os.path.join(outdir, f"{split}.csv")
            split = "train" if split == "dev" else split
            df[df.split == split].to_csv(outpath, index=False)
            ctx.log.info("wrote: %s", outpath)


def main(ctx: Context) -> None:
    default_outdir = os.path.join(
        dirparent(os.path.realpath(__file__), 2), "data", "proc", "preprocess"
    )
    formats = ["extraction", "inline"]
    versions = [v for v in cmu.Version]
    to_version = lambda s: cmu.Version[s]
    ctx.parser.add_argument("-f", "--format", choices=formats, required=True)
    ctx.parser.add_argument("-l", "--hlen", type=int, required=True)
    ctx.parser.add_argument("-s", "--seed", type=int, required=True)
    ctx.parser.add_argument(
        "-r", "--version", type=to_version, choices=versions, default="DEFAULT"
    )
    ctx.parser.add_argument("-o", "--outdir", default=default_outdir)
    args = ctx.parser.parse_args()

    # Process data set versions that use cross validation.
    if args.version.uses_cross_validation():
        return process_cross_validation(ctx, args)
    df = cmu.load(args.seed, version=args.version)
    # Print summary stats.
    summary = df.groupby(["split", "true_face"]).count().speaker.unstack(1)
    summary.index.name, summary.columns.name = None, None
    summary = summary.assign(total=summary.T.sum())
    summary.loc["total"] = summary.sum()
    list(map(ctx.log.info, str(summary).split("\n")))
    # Process to target format.
    fmap = {"extraction": preprocess_for_extraction, "inline": preprocess_for_inline}
    df = fmap[args.format](df, args.hlen)
    outdir = os.path.join(args.outdir, args.format, f"seed{args.seed}.hlen{args.hlen}")
    if args.version != cmu.Version.DEFAULT:
        outdir = f"{outdir}.with-{args.version.name.lower()}"
    os.makedirs(outdir, exist_ok=True)
    for split in ("train", "dev", "test"):
        outpath = os.path.join(outdir, f"{split}.csv")
        df[df.split == split].to_csv(outpath, index=False)
        ctx.log.info("wrote: %s", outpath)


if __name__ == "__main__":
    harness(main)
