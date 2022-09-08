"""
Train the models!

3 cases:
    * development/debugging/testing
    * hyper-parameter tuning
    * actual training
"""
import json
from dataclasses import dataclass
from typing import List
from typing import Type

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import get_config  # script-gen: config.py
from consts import ACCELERATOR  # script-gen: consts.py
from consts import DEVICES  # script-gen: consts.py
from consts import IS_LOCAL  # script-gen: consts.py
from consts import N_CLASSES  # script-gen: consts.py
from consts import OUTPUT_DATA_DIR  # script-gen: consts.py
from consts import PROCESSED_DATA_DIR  # script-gen: consts.py
from consts import RAW_DATA_DIR  # script-gen: consts.py
from data_module import DataModule  # script-gen: data_module.py
from file_reader import read_csv  # script-gen: file_reader.py
from logger import Logger  # script-gen: logger.py
from models import BaseModel  # script-gen: models.py
from models import BasicLinearModel  # noqa: F401 # script-gen: models.py
from models import ModelTools  # script-gen: logger.py
from prep_data import prep_data  # script-gen: prep_data.py

config = get_config()


@dataclass
class EvaluationResult:
    metrics: pd.DataFrame
    output_summary: pd.DataFrame

    def save(self):
        output_dir = OUTPUT_DATA_DIR / "evaluation"
        output_dir.mkdir(exist_ok=True)
        self.metrics.to_csv(output_dir / "metrics.csv", index=False)
        self.output_summary.to_csv(
            output_dir / "output_summary.csv",
            index=False,
        )


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
        self.ModelClass = ModelClass
        self.model_tools = model_tools
        self.logger = Logger()
        self.models: List[BaseModel] = []

    def fit(self) -> None:
        for fold in range(config["N_FOLDS"]):
            model = self.ModelClass(self.model_tools, {"fold": fold})
            datamodule = DataModule(fold)
            trainer = pl.Trainer(
                logger=self.logger,
                max_epochs=config["MAX_EPOCHS"],
                # We log after each epoch, this is just to silence a warning:
                log_every_n_steps=10,
                enable_checkpointing=False,
                accelerator=ACCELERATOR,
                devices=DEVICES,
            )
            trainer.fit(model, datamodule=datamodule)
            self.models.append(model)

    def make_submission(self) -> pd.DataFrame:
        assert (
            config["DATA"].stem == "full"
        ), "This function will not work on a partial dataset"
        submission = read_csv(RAW_DATA_DIR / "sample_submission.csv")
        eval_result = self.evaluate()
        probability_cols = [
            str(col)
            for col in eval_result.output_summary
            if str(col).startswith("prob_")
        ]
        predictions = (
            eval_result.output_summary.loc[
                eval_result.output_summary["label"] == -1
            ]
            .groupby("img_index")[probability_cols]
            .mean()
            .idxmax(axis=1)
            .apply(lambda label: int(label.split("_")[-1]))
        )
        assert len(predictions) == len(
            submission
        ), "Something went wrong, there are the wrong number of predictions"
        return submission.assign(Label=predictions)

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
    if not IS_LOCAL:
        prep_data(PROCESSED_DATA_DIR)

    print(
        "Training with config:",
        json.dumps({k: str(v) for k, v in config.items()}),
    )

    # TODO: no need for all of these to be passed in now. Instead, grab them
    # TODO: from config just before we need them
    mt = ModelTrainer(
        locals()[config["ARCHITECTURE"]],
        ModelTools(
            opt_class=getattr(torch.optim, config["OPTIM_CLASS"]),
            opt_args=config["OPTIM_PARAMS"],
            loss_func=getattr(torch.nn, config["LOSS_FUNC_CLASS"])(
                **config["LOSS_FUNC_PARAMS"]
            ),
        ),
    )

    mt.fit()
    if IS_LOCAL:
        result = mt.evaluate()
        result.save()
    else:
        submission = mt.make_submission()
        submission.to_csv(OUTPUT_DATA_DIR / "submission.csv", index=False)
