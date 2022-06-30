import sys
from pathlib import Path

import torch

# TODO: can this boilerplate be put in conftest.py or whatever?
sys.path.append(str(Path(__file__).parent.parent / "src"))
from models import BasicLinearModel
from models import ModelTools
from train import HyperparameterTrial


def test_loss_can_be_reduced():
    trial = HyperparameterTrial(
        BasicLinearModel,
        ModelTools(
            opt_class=torch.optim.SGD,
            opt_args={"lr": 1e-3},
            loss_func=torch.nn.functional.cross_entropy,
        ),
        fold=0,
        **{"max_epochs": 1},
    )

    loss_before = trial.sample_loss()
    trial.fit()
    loss_after = trial.sample_loss()
    assert loss_after < loss_before
