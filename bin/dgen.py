#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dialogue Generator"""
import asyncio
import datetime
import difflib
import hashlib
import os
from typing import Any, Iterable

import langchain as lc
import pandas as pd
from more_itertools import chunked, flatten

from src.core.app import harness
from src.core.context import Context
from src.core.path import dirparent
from src.core import keychain
from src.data import cmu, prompts


# TODO: Make this a function instead of global?
TEMPLATE = prompts.load("dgen", "one-shot")
CRITERIA = prompts.load("dgen", "criteria")


def get_example(cid: int, labeled: bool = True) -> str:
    df = cmu.load(42).query("conversation_id == @cid")
    assert len(df.conversation_id.unique()) == 1
    ret = []
    for _, row in df.iterrows():
        speaker = "EE" if row.speaker else "ER"
        if labeled:
            ret.append(f"{speaker}: {row.utterance} ({row.true_face})")
        else:
            ret.append(f"{speaker}: {row.utterance}")
    return "\n".join(ret)


def get_best_matches(
    word: str, possibilities: Iterable[str], n: int, cutoff: float
) -> list[str]:
    """
    Returns only the matches which have the maximum score.
    """
    matches = difflib.get_close_matches(word, possibilities, n=n, cutoff=cutoff)
    add_score = lambda w: (difflib.SequenceMatcher(a=word, b=w).ratio(), w)
    scores = list(map(add_score, matches))
    return [w for score, w in filter(lambda t: t[0] == max(scores)[0], scores)]


def repair_malformed_label(label: str) -> str:
    """
    Get the most similar labels up to a cutoff and returns the most frequent among them.
    """
    if label in cmu.FACEACTS:
        return label
    # Labels in descending order of frequency.
    order = ("other", "hpos+", "spos+", "hneg-", "hpos-", "hneg+", "sneg+", "spos-")
    matches = get_best_matches(label, cmu.FACEACTS, n=len(cmu.FACEACTS), cutoff=0.6)
    if not matches:
        return order[0]  # Return the most frequent label if there are no good matches.
    return sorted(matches, key=order.index)[0]


def process_dialogue(dlg: dict[str, Any]) -> list[dict[str, Any]]:
    ret = []
    lines = [l.strip() for l in dlg["generation"].split("\n") if l.strip()]
    for idx, line in enumerate(lines):
        speaker = line[:2]
        assert speaker in ("EE", "ER")
        label = repair_malformed_label(line.split(" ")[-1][1:-1])
        assert label in cmu.FACEACTS, dlg["generation"]
        utterance = " ".join(line.split(" ")[1:-1])
        ret.append(
            dlg
            | {"turn": idx, "turn_id": f"{dlg['conversation_id']}_{idx}"}
            | {"speaker": speaker, "utterance": utterance, "label": label}
        )
    return ret


async def get_dialogue(
    chat: lc.chat_models.ChatOpenAI, cid: int
) -> list[dict[str, Any]]:
    example = get_example(cid)
    prompt = TEMPLATE.format(criteria=CRITERIA, example=example)
    result = await chat.agenerate([[lc.schema.HumanMessage(content=prompt)]])
    assert len(result.generations) == 1 and len(result.generations[0]) == 1
    return process_dialogue(
        {
            "conversation_id": 1000 + cid,
            "prompt_conversation_id": cid,
            "prompt_template": TEMPLATE,
            "prompt_criteria": CRITERIA,
            "prompt_example": example,
            "prompt": prompt,
            "temperature": chat.temperature,
            "model_name": chat.model_name,
            "timestamp": datetime.datetime.now(),
            "generation": result.generations[0][0].text,
        }
    )


async def get_dialogues(
    chat: lc.chat_models.ChatOpenAI, cids: list[int]
) -> list[dict[str, Any]]:
    ret = []
    tasks = [get_dialogue(chat, cid) for cid in cids]
    for chunk in chunked(tasks, n=60):
        ret += await asyncio.gather(*chunk)
    return list(flatten(ret))


def main(ctx: Context) -> None:
    default_outdir = os.path.join(
        dirparent(os.path.realpath(__file__), 2), "data", "proc", "dgen"
    )
    ctx.parser.add_argument("-o", "--outdir", default=default_outdir)
    args = ctx.parser.parse_args()
    # Generate dialogues asynchronously.
    os.environ["OPENAI_API_KEY"] = keychain.get("OpenAI")
    chat = lc.chat_models.ChatOpenAI(model_name="gpt-3.5-turbo")  # type: ignore
    cids = list(cmu.load(42).conversation_id.unique())
    dlgs = pd.DataFrame(asyncio.run(get_dialogues(chat, cids)))
    # Write generated dialogues to file.
    prompt = TEMPLATE.format(criteria=CRITERIA, example="{example}")
    phash = hashlib.shake_256(prompt.encode("utf-8")).hexdigest(8)
    outname = datetime.datetime.now().strftime(f"{phash}.%Y%m%d.%H%M%S.csv")
    outpath = os.path.join(args.outdir, outname)
    os.makedirs(args.outdir, exist_ok=True)
    dlgs.to_csv(outpath, index=False)
    ctx.log.info("wrote: %s", outpath)


if __name__ == "__main__":
    harness(main)
