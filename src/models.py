"""
Start with the dumbest possible model
"""
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import Type

import pytorch_lightning as pl
import torch

from consts import INPUT_SIZE  # script-gen: consts.py
from consts import N_CLASSES  # script-gen: consts.py
from data_module import BatchType  # script-gen: data_module.py

LossFuncType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# TODO: What else should this class do? Is there a reason for it to exist?
@dataclass
class ModelTools:
    opt_class: Type
    opt_args: Dict[str, Any]
    loss_func: LossFuncType


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        tools: ModelTools,
        context: Dict[str, Any] = {},
    ) -> None:
        super().__init__()
        self.tools = tools
        self.context = context

    def training_step(  # type: ignore
        self,
        batch: BatchType,
        batch_idx: int,
    ) -> torch.Tensor:
        x, y, *_ = batch
        pred = self(x)
        loss = self.tools.loss_func(pred, y)
        accuracy = (pred.argmax(dim=1) == y).float().mean()
        self.log_dict(
            {"train_loss": loss, "train_accuracy": accuracy, **self.context},
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(  # type: ignore
        self,
        batch: BatchType,
        batch_idx: int,
    ) -> torch.Tensor:
        x, y, *_ = batch
        pred = self(x)
        loss = self.tools.loss_func(pred, y)
        accuracy = (pred.argmax(dim=1) == y).float().mean()
        self.log_dict(
            {"val_loss": loss, "val_accuracy": accuracy, **self.context},
            on_step=False,
            on_epoch=True,
        )
        return loss

    def configure_optimizers(self):
        return self.tools.opt_class(self.parameters(), **self.tools.opt_args)


class AlwaysSayZeroModel(BaseModel):
    """An extremely wrong but simple model for easy testing"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return torch.zeros((len(x), N_CLASSES))


class BasicLinearModel(BaseModel):
    """Something simple but trainable"""

    def __init__(
        self,
        tools: ModelTools,
        context: Dict[str, Any] = {},
    ) -> None:
        super().__init__(tools, context)
        hidden_layer_size = INPUT_SIZE // 2
        self.net = torch.nn.Sequential(
            torch.nn.Linear(INPUT_SIZE, hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_size, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.net.forward(x)
