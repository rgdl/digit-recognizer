from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd  # type: ignore
import pytorch_lightning as pl
import torch

from models import BasicLinearModel
from models import ModelTools
from train import ModelTrainer


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
            **{"max_epochs": 3, "logger": pl.loggers.CSVLogger(td)},
        )

        mt.fit()
        logs_df = pd.read_csv(Path(td, "lightning_logs/version_0/metrics.csv"))
        val_loss_logs = logs_df["val_loss"].dropna()
        first_val_loss = val_loss_logs.iloc[0]
        last_val_loss = val_loss_logs.iloc[-1]
        assert last_val_loss < first_val_loss
