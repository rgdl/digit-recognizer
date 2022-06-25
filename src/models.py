"""
Start with the dumbest possible model
"""
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import Tuple

import pytorch_lightning as pl
import torch

from consts import INPUT_SIZE
from consts import N_CLASSES

BatchType = Tuple[torch.Tensor, torch.Tensor]
LossFuncType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class ModelTools:
    opt_class: torch.optim.Optimizer
    opt_args: Dict[str, Any]
    loss_func: LossFuncType


class BaseModel(pl.LightningModule):
    def __init__(self, tools: ModelTools) -> None:
        super().__init__()
        self.tools = tools

    def training_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        x, y = batch
        pred = self(x)
        loss = self.tools.loss_func(pred, y)
        return loss
    
    def configure_optimizers(self):
        return self.tools.opt_class(self.parameters(), **self.tools.opt_args)


class AlwaysSayZeroModel(BaseModel):
    """An extremely wrong but simple model for easy testing"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((len(x), N_CLASSES))


class BasicLinearModel(BaseModel):
    """
    Something simple but trainable
    """
    def __init__(self, tools: ModelTools) -> None:
        super().__init__(tools)
        hidden_layer_size = INPUT_SIZE // 2
        self.net = torch.nn.Sequential(
            torch.nn.Linear(INPUT_SIZE, hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_size, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.forward(x)