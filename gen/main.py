"""
Train the models!

3 cases:
    * development/debugging/testing
    * hyper-parameter tuning
    * actual training
"""
from pathlib import Path
from typing import Any
from typing import Type

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

### Contents of 'consts.py' ###
from pathlib import Path
from typing import Any
from typing import Dict

_consts = None


def get_consts() -> Dict[str, Any]:
    """
    Wrapped into a function so elements can be over-ridden in tests, cached at
    the module level to get singleton behaviour
    """
    global _consts
    if _consts is None:
        ROOT_DIR = Path(__file__).parent.parent
        DATA_DIR = ROOT_DIR / "data"
        RAW_DATA_DIR = DATA_DIR / "raw"
        PROCESSED_DATA_DIR = DATA_DIR / "processed"
        OUTPUT_DATA_DIR = DATA_DIR / "output"

        DATA = PROCESSED_DATA_DIR / "mini.pickle"

        assert DATA.exists()

        _consts = {
            # Dataset properties
            "TEST_DATA_ROWS": 28000,
            "TRAIN_DATA_ROWS": 42000,
            "INPUT_SIZE": 784,
            "N_CLASSES": 10,
            # Training parameters
            "N_WORKERS": 4,
            "BATCH_SIZE": 32,
            # Data prep
            "N_FOLDS": 5,
            "MINI_DATA_PROPORTION": 0.1,
            "MICRO_DATA_PROPORTION": 0.01,
            # File paths
            "ROOT_DIR": ROOT_DIR,
            "DATA_DIR": DATA_DIR,
            "RAW_DATA_DIR": RAW_DATA_DIR,
            "PROCESSED_DATA_DIR": PROCESSED_DATA_DIR,
            "OUTPUT_DATA_DIR": OUTPUT_DATA_DIR,
            "DATA": DATA,
            # Random seed
            "SEED": 777,
        }
    return _consts
### End of 'consts.py' ###
### Contents of 'data_module.py' ###
from typing import Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


consts = get_consts()
BatchType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class Dataset(BaseDataset):
    def __init__(self, data: pd.DataFrame) -> None:
        self.x = torch.FloatTensor(
            data[
                [
                    col
                    for col in data
                    if isinstance(col, str) and col.startswith("pixel")
                ]
            ].values
        )
        self.y = torch.LongTensor(data["label"].values)
        self.ind = torch.IntTensor(data.index)
        self._len = len(self.x)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> BatchType:
        return self.x[idx], self.y[idx], self.ind[idx]


class DataModule(pl.LightningDataModule):
    COMMON_DATA_LOADER_ARGS = {
        "batch_size": consts["BATCH_SIZE"],
        "num_workers": consts["N_WORKERS"],
        "drop_last": False,
    }

    def __init__(self, fold: int):
        super().__init__()
        self.fold = fold
        self.all_data = pd.read_pickle(consts["DATA"])

        assert 0 <= fold <= self.all_data["fold"].max()

        self._test_rows = self.all_data["fold"] < 0
        self._valid_rows = self.all_data["fold"] == fold
        self._train_rows = ~(self._test_rows | self._valid_rows)

    def train_dataloader(self):
        data = self.all_data.loc[self._train_rows].drop("fold", axis=1)
        ds = Dataset(data)
        return DataLoader(
            ds,
            **self.COMMON_DATA_LOADER_ARGS,  # type: ignore
            shuffle=True,
        )

    def val_dataloader(self):
        data = self.all_data.loc[self._valid_rows].drop("fold", axis=1)
        ds = Dataset(data)
        return DataLoader(ds, **self.COMMON_DATA_LOADER_ARGS)  # type: ignore

    def test_dataloader(self):
        data = self.all_data.loc[self._test_rows].drop("fold", axis=1)
        ds = Dataset(data)
        return DataLoader(ds, **self.COMMON_DATA_LOADER_ARGS)  # type: ignore
### End of 'data_module.py' ###
### Contents of 'models.py' ###
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


LossFuncType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
consts = get_consts()


@dataclass
class ModelTools:
    opt_class: Type
    opt_args: Dict[str, Any]
    loss_func: LossFuncType


class BaseModel(pl.LightningModule):
    def __init__(self, tools: ModelTools) -> None:
        super().__init__()
        self.tools = tools

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
            {"train_loss": loss, "train_accuracy": accuracy},
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
            {"val_loss": loss, "val_accuracy": accuracy},
            on_step=False,
            on_epoch=True,
        )
        return loss

    def configure_optimizers(self):
        return self.tools.opt_class(self.parameters(), **self.tools.opt_args)


class AlwaysSayZeroModel(BaseModel):
    """An extremely wrong but simple model for easy testing"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return torch.zeros((len(x), consts["N_CLASSES"]))


class BasicLinearModel(BaseModel):
    """Something simple but trainable"""

    def __init__(self, tools: ModelTools) -> None:
        super().__init__(tools)
        hidden_layer_size = consts["INPUT_SIZE"] // 2
        self.net = torch.nn.Sequential(
            torch.nn.Linear(consts["INPUT_SIZE"], hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_size, consts["N_CLASSES"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.net.forward(x)
### End of 'models.py' ###

consts = get_consts()


# TODO: make use of pytorch lightning's inbuilt HP tuning stuff
class ModelTrainer:
    def __init__(
        self,
        ModelClass: Type,
        model_tools: ModelTools,
        fold: int,
        **trainer_kwargs: Any,
    ) -> None:
        """
        Train `model` with `model_tools` and return the validation loss
        Hyperparameter tuning will then aim to minimise this value.
        """
        self.data = DataModule(fold)
        self.model = ModelClass(model_tools)
        self.trainer = pl.Trainer(
            **trainer_kwargs,
            # We log after each epoch, just setting this to silence a warning:
            log_every_n_steps=10,
        )

    def fit(self) -> None:
        self.trainer.fit(self.model, datamodule=self.data)

    def evaluate(self) -> None:
        if isinstance(self.trainer.logger, pl.loggers.CSVLogger):
            self.get_output_summary().to_csv(
                Path(self.trainer.logger.log_dir, "output_summary.csv"),
                index=False,
            )
        else:
            raise NotImplementedError

    def _get_dataloader(self, dataset: str) -> torch.utils.data.DataLoader:
        dataset_dataloader_map = {
            "train": self.data.train_dataloader(),
            "valid": self.data.val_dataloader(),
            "test": self.data.test_dataloader(),
        }
        try:
            return dataset_dataloader_map[dataset]
        except IndexError:
            raise ValueError(
                "`dataset` must be one of "
                f"{list(dataset_dataloader_map.keys())}"
            )

    def get_output_summary(self) -> pd.DataFrame:
        # TODO: the output should link back to input files
        probability_cols = [f"prob_{i}" for i in range(consts["N_CLASSES"])]

        def _get_dataset_summary(dataset: str) -> pd.DataFrame:
            inds, preds, labels = [], [], []
            with torch.no_grad():
                for x, y, ind in self._get_dataloader(dataset):
                    preds.append(self.model(x).cpu().detach())
                    labels.append(y.cpu().detach())
                    inds.extend(ind.tolist())
            all_preds = F.softmax(torch.concat(preds), dim=1).numpy()
            all_labels = torch.concat(labels).numpy()

            result = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "img_index": inds,
                            "label": all_labels,
                            "is_valid": dataset == "valid",
                        }
                    ),
                    pd.DataFrame(all_preds, columns=probability_cols),
                ],
                axis=1,
            )
            result["pred"] = result.filter(regex="prob").values.argmax(axis=1)
            result["correct"] = result["pred"] == result["label"]
            return result

        return pd.concat(
            [
                _get_dataset_summary("train"),
                _get_dataset_summary("valid"),
                _get_dataset_summary("test"),
            ]
        )


if __name__ == "__main__":

    mt = ModelTrainer(
        BasicLinearModel,
        ModelTools(
            # TODO: bundle into OptimiserArgs
            opt_class=torch.optim.SGD,
            opt_args={"lr": 1e-3},
            loss_func=torch.nn.functional.cross_entropy,
        ),
        fold=0,
        **{
            "max_epochs": 10,
            "logger": pl.loggers.CSVLogger(str(consts["OUTPUT_DATA_DIR"])),
        },
    )

    mt.fit()
    mt.evaluate()
