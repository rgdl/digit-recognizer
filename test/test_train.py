import pandas as pd
import pytest
import torch

from consts import get_consts
from models import BasicLinearModel
from models import ModelTools
from train import ModelTrainer

consts = get_consts()


@pytest.fixture(scope="session")
def trained_model():
    # Override config for fast but effective training
    consts["DATA"] = consts["PROCESSED_DATA_DIR"] / "micro.pickle"
    consts["MAX_EPOCHS"] = 2
    consts["N_FOLDS"] = 1

    mt = ModelTrainer(
        BasicLinearModel,
        ModelTools(
            opt_class=torch.optim.SGD,
            opt_args={"lr": 1e-3},
            loss_func=torch.nn.functional.cross_entropy,
        ),
    )

    mt.fit()
    return mt


def test_loss_can_be_reduced(trained_model):
    train_loss = [
        item["train_loss"]
        for item in trained_model.logger.metrics
        if "train_loss" in item
    ]
    assert (
        train_loss[-1] < train_loss[0]
    ), "Try running again, as this is a potentially flaky test"


def test_output_summary(trained_model):
    df = trained_model.evaluate().output_summary

    assert df.notna().all(axis=None)  # type: ignore

    assert tuple(df.columns) == (
        "img_index",
        "label",
        "is_valid",
        *(f"prob_{i}" for i in range(consts["N_CLASSES"])),
        "pred",
        "correct",
        "fold",
    )
    assert len(df) == len(pd.read_pickle(consts["DATA"]))

    assert df["label"].dtype == int
    assert df["label"].min() == -1
    assert df["label"].max() == 9

    assert df["is_valid"].dtype == bool

    for col in df.filter(regex="^prob_[0-9]$"):
        assert df[col].dtype == "float32"
        assert df[col].min() >= 0
        assert df[col].max() <= 1

    assert df["pred"].dtype == int
    assert df["pred"].min() >= 0
    assert df["pred"].max() <= 9

    assert df["correct"].dtype == bool

    assert df["fold"].dtype == int
    # This is the current training fold, so never -1
    assert df["fold"].min() == 0
    assert df["fold"].max() == consts["N_FOLDS"] - 1
