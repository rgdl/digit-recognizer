"""
Take in the outputs of a local training run and analyse

We want to know:
    * items with highest loss (i.e. most confidently wrong)
    * confusion matrix
    * movement of train and validation loss during training
"""
import math
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore

from config import get_config

config = get_config()


def subplotter(title: str):
    """
    Encapsulate some of the messy logic with matplotlib's subplots
    """
    fig, ax = plt.subplots(2, math.ceil(config["N_FOLDS"] / 2))
    fig.suptitle(title)

    rows, cols = ax.shape
    for row in range(rows):
        for col in range(cols):
            yield ax[row][col]


class Analyser:
    def __init__(self, metrics: pd.DataFrame, outputs: pd.DataFrame) -> None:
        self.metrics = metrics
        self.outputs = outputs
        self.data = pd.read_pickle(config["DATA"])

    def view_training_metrics(self, output_file: Optional[str] = None) -> None:
        axes = subplotter("Accuracy Values During Training")
        for fold in range(config["N_FOLDS"]):
            ax = next(axes)
            for col in ("train_accuracy", "val_accuracy"):
                plot_df = self.metrics.dropna(subset=[col])
                plot_df = plot_df.loc[plot_df["fold"] == fold]
                ax.plot(plot_df["step"], plot_df[col], label=col)
                ax.set_title(f"Fold {fold}")
            if fold == 0:
                ax.legend()
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()

    def get_confusion_matrix(self):
        conf_mat = (
            self.outputs.loc[self.outputs["is_valid"], ["label", "pred"]]
            .assign(count=1)
            .groupby(["label", "pred"], as_index=False)["count"]
            .count()
            .pivot("label", "pred", "count")
            .fillna(0)
            .astype(int)
        )
        assert conf_mat.shape[0] == conf_mat.shape[1]
        return conf_mat

    def view_confusion_matrix(self, hide_diagonals: bool = False):
        cm = self.get_confusion_matrix()
        if hide_diagonals:
            np.fill_diagonal(cm.values, 0)
        sns.heatmap(cm)
        plt.show()

    def view_image(self):
        pass

    def most_confidently_wrong(self, n: int = 5):
        idx = (
            self.outputs.loc[
                ~self.outputs["correct"] & self.outputs["is_valid"]
            ]
            .filter(regex="prob_")
            .max(axis=1)
            .sort_values(ascending=False)
            .head(n)
            .index
        )
        most_confused = self.outputs.loc[idx].merge(
            self.data.drop(["label", "fold"], axis=1),
            left_on="img_index",
            right_index=True,
        )
        for _, row in most_confused.iterrows():
            pixels = (
                row.filter(regex="pixel").values.astype(float).reshape(28, 28)
            )
            plt.imshow(pixels)
            label = row["label"]
            pred = row["pred"]
            conf = row[f"prob_{pred}"]
            plt.title(
                "\n".join(
                    [
                        f"Label: {label}, Pred: {pred} ({100 * conf:.2f}%)",
                        f"Fold {row['fold']}",
                    ]
                )
            )
            plt.show()
        return most_confused


# TODO: method to run other functions to create an html report with the images


class LogDir:
    def __init__(self, path: str) -> None:
        log_dir = Path(sys.argv[1])
        assert log_dir.exists(), "The provided log directory does not exist"
        self.metrics = pd.read_csv(log_dir / "metrics.csv")
        self.outputs = pd.read_csv(log_dir / "output_summary.csv")


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please provide a log directory"
    log_dir = LogDir(sys.argv[1])

    analyser = Analyser(log_dir.metrics, log_dir.outputs)
    analyser.view_training_metrics()
    analyser.view_confusion_matrix(hide_diagonals=True)
    analyser.most_confidently_wrong()
