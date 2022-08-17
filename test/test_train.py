from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd  # type: ignore
import pytorch_lightning as pl
import torch

from consts import N_CLASSES
from models import BasicLinearModel
from models import ModelTools
from train import ModelTrainer

# TODO: create a fixture for a model trainer excluding the case where we make
# TODO: a temp dir to store the logs, IF WE HAVE TO


def test_loss_can_be_reduced():
    with TemporaryDirectory() as td:
        mt = ModelTrainer(
            BasicLinearModel,
            ModelTools(
                opt_class=torch.optim.SGD,
                opt_args={"lr": 1e-3},
                loss_func=torch.nn.functional.cross_entropy,
            ),
            fold=0,
            **{"max_epochs": 5, "logger": pl.loggers.CSVLogger(td)},
        )

        mt.fit()
        logs_df = pd.read_csv(Path(td, "lightning_logs/version_0/metrics.csv"))
        val_loss_logs = logs_df["val_loss"].dropna()
        first_val_loss = val_loss_logs.iloc[0]
        last_val_loss = val_loss_logs.iloc[-1]
        assert last_val_loss < first_val_loss


datasets_and_dataloaders = [
    ("train", "train_dataloader"),
    ("valid", "val_dataloader"),
    ("test", "test_dataloader"),
]


def test_get_output_summary():
    mt = ModelTrainer(
        BasicLinearModel,
        ModelTools(
            opt_class=torch.optim.SGD,
            opt_args={"lr": 1e-3},
            loss_func=torch.nn.functional.cross_entropy,
        ),
        fold=0,
        **{"max_epochs": 3},
    )
    df = mt.get_output_summary()

    assert df.notna().all(axis=None)  # type: ignore

    assert tuple(df.columns) == (
        "label",
        "is_valid",
        *(f"prob_{i}" for i in range(N_CLASSES)),
        "pred",
        "correct",
    )
    assert len(df) == (
        len(mt.data.train_dataloader().dataset)
        + len(mt.data.val_dataloader().dataset)
        + len(mt.data.test_dataloader().dataset)
    )

    assert df["label"].dtype == int
    assert df["label"].min() == -1
    assert df["label"].max() == 9

    assert df["is_valid"].dtype == bool

    for col in df.filter(regex="^prob_[0-9]$"):
        assert df[col].dtype == "float32"
        assert df[col].min() >= 0
        assert df[col].max() <= 1

    assert df["pred"].dtype == int
    assert df["pred"].min() == 0
    assert df["pred"].max() == 9

    assert df["correct"].dtype == bool
