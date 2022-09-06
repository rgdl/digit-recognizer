import pytest
import torch

from config import get_config
from consts import N_CLASSES
from consts import PROCESSED_DATA_DIR
from consts import RAW_DATA_DIR
from file_reader import read_csv
from file_reader import read_pickle
from models import BasicLinearModel
from models import ModelTools
from train import ModelTrainer

config = get_config()


@pytest.fixture(scope="session")
def untrained_model():
    # Override config for no training at all, but using full dataset
    config["DATA"] = PROCESSED_DATA_DIR / "full.pickle"
    config["MAX_EPOCHS"] = 0
    config["N_FOLDS"] = 1

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


@pytest.fixture(scope="session")
def trained_model():
    # Override config for fast but effective training
    config["DATA"] = PROCESSED_DATA_DIR / "micro.pickle"
    config["MAX_EPOCHS"] = 2
    config["N_FOLDS"] = 1

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
        *(f"prob_{i}" for i in range(N_CLASSES)),
        "pred",
        "correct",
        "fold",
    )
    assert len(df) == len(read_pickle(config["DATA"]))

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
    assert df["fold"].max() == config["N_FOLDS"] - 1


def test_make_submission(untrained_model):
    submission = untrained_model.make_submission()
    sample_submission = read_csv(RAW_DATA_DIR / "sample_submission.csv")
    assert tuple(submission.columns) == ("ImageId", "Label")
    assert len(submission) == len(sample_submission)
    assert (submission["ImageId"] == sample_submission["ImageId"]).all()
    assert sample_submission["Label"].isin(range(0, 10)).all()
