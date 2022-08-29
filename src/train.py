"""
Train the models!

3 cases:
    * development/debugging/testing
    * hyper-parameter tuning
    * actual training
"""
from dataclasses import dataclass
from typing import List
from typing import Type

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import get_config  # script-gen: config.py
from consts import N_CLASSES  # script-gen: consts.py
from consts import OUTPUT_DATA_DIR  # script-gen: consts.py
from data_module import DataModule  # script-gen: data_module.py
from logger import Logger  # script-gen: models.py
from models import BaseModel  # script-gen: logger.py
from models import ModelTools  # script-gen: logger.py

config = get_config()


@dataclass
class EvaluationResult:
    metrics: pd.DataFrame
    output_summary: pd.DataFrame

    def save(self):
        output_dir = OUTPUT_DATA_DIR / "evaluation"
        output_dir.mkdir(exist_ok=True)
        self.metrics.to_csv(output_dir / "metrics.csv")
        self.output_summary.to_csv(output_dir / "output_summary.csv")


class ModelTrainer:
    def __init__(
        self,
        ModelClass: Type,
        model_tools: ModelTools,
    ) -> None:
        """
        Train `model` with `model_tools` and return the validation loss
        Hyperparameter tuning will then aim to minimise this value.
        """
        # TODO: get modelclass and model tools from config/config
        self.ModelClass = ModelClass
        self.model_tools = model_tools
        self.logger = Logger()
        self.models: List[BaseModel] = []

    def fit(self) -> None:
        for fold in range(config["N_FOLDS"]):
            model = self.ModelClass(self.model_tools)
            datamodule = DataModule(fold)
            trainer = pl.Trainer(
                logger=self.logger,
                max_epochs=config["MAX_EPOCHS"],
                # We log after each epoch, this is just to silence a warning:
                log_every_n_steps=10,
                enable_checkpointing=False,
            )
            trainer.fit(model, datamodule=datamodule)
            self.models.append(model)

    def _get_dataset_summary(
        self, model: BaseModel, dataloader: DataLoader, is_valid: bool = False
    ) -> pd.DataFrame:
        inds, preds, labels = [], [], []
        with torch.no_grad():
            for x, y, ind in dataloader:
                preds.append(model(x).cpu().detach())
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
                        "is_valid": is_valid,
                    }
                ),
                pd.DataFrame(
                    all_preds,
                    columns=[f"prob_{i}" for i in range(N_CLASSES)],
                ),
            ],
            axis=1,
        )
        result["pred"] = result.filter(regex="prob").values.argmax(axis=1)
        result["correct"] = result["pred"] == result["label"]
        return result

    def evaluate(self) -> EvaluationResult:
        output_summary_dfs = []
        for model, fold in zip(self.models, range(config["N_FOLDS"])):
            datamodule = DataModule(fold)
            output_summary_dfs.append(
                pd.concat(
                    [
                        self._get_dataset_summary(
                            model, datamodule.train_dataloader()
                        ),
                        self._get_dataset_summary(
                            model, datamodule.val_dataloader(), True
                        ),
                        self._get_dataset_summary(
                            model, datamodule.test_dataloader()
                        ),
                    ]
                ).assign(fold=fold)
            )
        return EvaluationResult(
            metrics=pd.DataFrame.from_records(self.logger.metrics),
            output_summary=pd.concat(output_summary_dfs),
        )


if __name__ == "__main__":
    from models import BasicLinearModel  # script-gen: models.py

    mt = ModelTrainer(
        BasicLinearModel,
        ModelTools(
            # TODO: bundle into OptimiserArgs
            opt_class=torch.optim.SGD,
            opt_args={"lr": 1e-3},
            loss_func=torch.nn.functional.cross_entropy,
        ),
    )

    mt.fit()
    result = mt.evaluate()
    result.save()
