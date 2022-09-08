"""
Tune hyperparameters using optuna
"""
import pickle

import optuna
import torch

from config import get_config  # script-gen: config.py
from consts import IS_LOCAL  # script-gen: consts.py
from consts import OUTPUT_DATA_DIR  # script-gen: consts.py
from consts import PROCESSED_DATA_DIR  # script-gen: consts.py
from models import BasicLinearModel  # noqa: F401 # script-gen: models.py
from models import ModelTools  # script-gen: models.py
from prep_data import prep_data  # script-gen: prep_data.py
from train import ModelTrainer  # script-gen: train.py

config = get_config()


def objective(trial):
    config["N_FOLDS"] = 1
    config["ARCHITECTURE"] = trial.suggest_categorical(
        "architecture", ("BasicLinearModel",)
    )
    config["MAX_EPOCHS"] = trial.suggest_int("max_epochs", 1, 20)

    config["OPTIM_CLASS"] = trial.suggest_categorical("optim_class", ("SGD",))
    config["OPTIM_PARAMS"] = {}
    if config["OPTIM_CLASS"] == "SGD":
        config["OPTIM_PARAMS"]["lr"] = trial.suggest_float(
            "optim_params.lr", 1e-6, 1, log=True
        )

    config["LOSS_FUNC_CLASS"] = trial.suggest_categorical(
        "loss_func_class", ("CrossEntropyLoss",)
    )
    # Not currently selecting params for loss function classes

    mt = ModelTrainer(
        globals()[config["ARCHITECTURE"]],
        ModelTools(
            opt_class=getattr(torch.optim, config["OPTIM_CLASS"]),
            opt_args=config["OPTIM_PARAMS"],
            loss_func=getattr(torch.nn, config["LOSS_FUNC_CLASS"])(
                **config["LOSS_FUNC_PARAMS"]
            ),
        ),
    )

    mt.fit()
    result = mt.evaluate()

    return result.metrics.groupby("fold")["val_loss"].last().mean()


if __name__ == "__main__":
    if not IS_LOCAL:
        prep_data(PROCESSED_DATA_DIR)

    # TODO: pruning? Maybe if gradients explode or collapse?
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    with open(OUTPUT_DATA_DIR / "tune.pickle", "wb") as f:
        pickle.dump(study, f)
