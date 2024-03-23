# -*- coding: utf-8 -*
import enum
import os
from typing import Tuple

import numpy as np
import pandas as pd

from ..core.path import dirparent


FACEACTS = ("sneg-", "spos+", "hpos+", "spos-", "hpos-", "sneg+", "hneg+", "hneg-", "other")
RAW_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "raw")
RAW_PATH = os.path.join(RAW_DIR, "Persuasion-Face-Act-Prediction.csv")
POL_PATH = os.path.join(RAW_DIR, "Persuasion-Face-Act-Prediction.With-Politeness.csv")
MRDA_PATH = os.path.join(RAW_DIR, "Persuasion-Face-Act-Prediction.With-MRDA.csv")
SWDA_PATH = os.path.join(RAW_DIR, "Persuasion-Face-Act-Prediction.With-SWDA.csv")
XVAL_PATH = os.path.join(RAW_DIR, "Persuasion-Face-Act-Prediction-XVAL.csv")
XVAL_MRDA_PATH = os.path.join(RAW_DIR, "Persuasion-Face-Act-Prediction-XVAL.With-MRDA.csv")
XVAL_SWDA_PATH = os.path.join(RAW_DIR, "Persuasion-Face-Act-Prediction-XVAL.With-SWDA.csv")
MTL_SWDA_PATH = os.path.join(RAW_DIR, "Persuasion-Face-Act-Prediction-MTL.With-SWDA.csv")
MTL_MRDA_PATH = os.path.join(RAW_DIR, "Persuasion-Face-Act-Prediction-MTL.With-MRDA.csv")
MTL_MRDA_ALL_PATH = os.path.join(RAW_DIR, "Persuasion-Face-Act-Prediction-MTL-ALL.With-MRDA.csv")
MTL_OASIS_PATH = os.path.join(RAW_DIR, "Persuasion-Face-Act-Prediction-MTL.With-OASIS.csv")
MTL_POL_PATH = os.path.join(RAW_DIR, "Persuasion-Face-Act-Prediction-MTL.With-POL.csv")


class Version(enum.Enum):
    DEFAULT = RAW_PATH
    POLITENESS = POL_PATH
    MRDA = MRDA_PATH
    SWDA = SWDA_PATH
    XVAL = XVAL_PATH
    XVAL_MRDA = XVAL_MRDA_PATH
    XVAL_SWDA = XVAL_SWDA_PATH
    MTL_SWDA = MTL_SWDA_PATH
    MTL_MRDA = MTL_MRDA_PATH          # Balanced 1 to 1.
    MTL_MRDA_ALL = MTL_MRDA_ALL_PATH  # All data.
    MTL_OASIS = MTL_OASIS_PATH
    MTL_POL = MTL_POL_PATH

    def uses_cross_validation(self) -> bool:
        return self.name.startswith("XVAL") or self.name.startswith("MTL")

    def uses_mtl(self) -> bool:
        return self.name.startswith("MTL")


def train_dev_test_split(
    data: pd.DataFrame, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Split files 70/10/20.
    cids = sorted(data.conversation_id.unique())
    np.random.seed(seed)
    np.random.shuffle(cids)
    train, dev, test = np.split(cids, [int(len(cids) * 0.7), int(len(cids) * 0.8)])
    return (
        data[data.conversation_id.isin(train)],
        data[data.conversation_id.isin(dev)],
        data[data.conversation_id.isin(test)],
    )


# XXX: Unused.
def cross_validation_split(data: pd.DataFrame, seed: int) -> pd.DataFrame:
    ret = []
    # Folds are there just need to add a dev set.
    for fold in sorted(data.fold.unique()):
        df = data[data.fold == fold]
        assert set(df.split.unique()) == {"train", "test"}
        # Split a dev set out from the train set.
        train, test = df[df.split == "train"], df[df.split == "test"]
        cids = sorted(train.conversation_id.unique())
        np.random.seed(seed)
        np.random.shuffle(cids)
        train_cids, dev_cids = np.split(cids, [int(len(cids) * 0.9)])
        dev = df[df.conversation_id.isin(dev_cids)]
        train = df[df.conversation_id.isin(train_cids)]
        # Check nothing went wrong.
        assert len(train) + len(dev) + len(test) == len(df)
        assert train.split.unique().tolist() == ["train"]
        assert dev.split.unique().tolist() == ["train"]
        assert test.split.unique().tolist() == ["test"]
        ret.append(pd.concat([train, dev.assign(split="dev"), test]))
    ret = pd.concat(ret)
    assert len(ret) == len(data)
    return ret


def load(seed: int, version: Version = Version.DEFAULT) -> pd.DataFrame:
    if version.uses_cross_validation():
        return pd.read_csv(version.value)
    train, dev, test = train_dev_test_split(pd.read_csv(version.value), seed=seed)
    return pd.concat(
        [
            train.assign(split="train"),
            dev.assign(split="dev"),
            test.assign(split="test"),
        ]
    )
