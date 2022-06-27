import sys
from pathlib import Path

import pytorch_lightning as pl
import torch

# TODO: can this boilerplate be put in conftest.py or whatever?
sys.path.append(str(Path(__file__).parent.parent / "src"))
from data_module import DataModule
from models import BasicLinearModel
from models import ModelTools


def test_loss_can_be_reduced():
    data = DataModule(0)
    model = BasicLinearModel(
        ModelTools(
            opt_class=torch.optim.SGD,
            opt_args={"lr": 1e-3},
            loss_func=torch.nn.functional.cross_entropy,
        )
    )
    batch = next(iter(data.train_dataloader()))
    loss_before = model.training_step(batch, 0)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, datamodule=data)
    loss_after = model.training_step(batch, 0)

    assert loss_after < loss_before
