import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent / "src"))
from consts import BATCH_SIZE
from consts import N_CLASSES
from data_module import DataModule
from models import AlwaysSayZeroModel
from models import BasicLinearModel
from models import ModelTools

all_models = (
    AlwaysSayZeroModel,
    BasicLinearModel,
)


def test_output_shape():
    dm = DataModule(fold=0)
    model_tools = ModelTools(
        opt_class=torch.optim.SGD,
        opt_args={"lr": 1e-3},
        loss_func=torch.nn.functional.cross_entropy,
    )
    x, _ = next(iter(dm.train_dataloader()))
    for Model in all_models:
        model = Model(model_tools)
        pred = model(x)
        assert pred.shape == (BATCH_SIZE, N_CLASSES)
