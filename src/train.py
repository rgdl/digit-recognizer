"""
Train the models!

3 cases:
    * development/debugging/testing
    * hyper-parameter tuning
    * actual training
"""
from typing import Any
from typing import Type

import pandas as pd  # type: ignore
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from consts import N_CLASSES
from consts import OUTPUT_DATA_DIR
from data_module import DataModule
from models import ModelTools


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
        self.trainer = pl.Trainer(**trainer_kwargs)  # type: ignore

    def fit(self) -> None:
        self.trainer.fit(self.model, datamodule=self.data)

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

    # TODO: Change to just predict test (hard-coded)
    def batch_predict(self, dataset: str) -> torch.Tensor:
        preds = []
        with torch.no_grad():
            for x, _ in self._get_dataloader(dataset):
                preds.append(self.model(x).cpu().detach())
        return torch.concat(preds)

    def get_output_summary(self) -> pd.DataFrame:
        probability_cols = [f"prob_{i}" for i in range(N_CLASSES)]

        def _get_dataset_summary(dataset: str) -> pd.DataFrame:
            preds, labels = [], []
            with torch.no_grad():
                for x, y in self._get_dataloader(dataset):
                    preds.append(self.model(x).cpu().detach())
                    labels.append(y.cpu().detach())
            all_preds = F.softmax(torch.concat(preds), dim=1).numpy()
            all_labels = torch.concat(labels).numpy()

            result = pd.concat(
                [
                    pd.DataFrame(
                        {
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
            ]
        )


if __name__ == "__main__":
    from models import BasicLinearModel

    mt = ModelTrainer(
        BasicLinearModel,
        ModelTools(
            opt_class=torch.optim.SGD,
            opt_args={"lr": 1e-3},
            loss_func=torch.nn.functional.cross_entropy,
        ),
        fold=0,
        **{
            "max_epochs": 5,
            "logger": pl.loggers.CSVLogger(str(OUTPUT_DATA_DIR)),
        },
    )

    mt.fit()
