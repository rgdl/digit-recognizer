"""
Train the models!

3 cases:
    * development/debugging/testing
    * hyper-parameter tuning
    * actual training
"""
from typing import Any
from typing import Dict
from typing import Type

import pytorch_lightning as pl

from data_module import DataModule
from models import ModelTools


class ModelTrainer:
    def __init__(
        self,
        ModelClass: Type,
        model_tools: ModelTools,
        fold: int,
        **trainer_kwargs: Dict[str, Any],
    ) -> None:
        """
        Train `model` with `model_tools` and return the validation loss
        Hyperparameter tuning will then aim to minimise this value.
        """
        self.data = DataModule(fold)
        self.model = ModelClass(model_tools)
        self.trainer = pl.Trainer(**trainer_kwargs)  # type: ignore

    def fit(self) -> None:
        self.trainer.fit(self.model, datamodule=self.data)
