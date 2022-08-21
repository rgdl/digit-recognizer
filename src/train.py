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

from consts import get_consts
from data_module import DataModule
from models import ModelTools

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
    from models import BasicLinearModel

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
            "max_epochs": 5,
            "logger": pl.loggers.CSVLogger(str(consts["OUTPUT_DATA_DIR"])),
        },
    )

    mt.fit()
    mt.evaluate()
