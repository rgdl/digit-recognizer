"""
Take in the outputs of a local training run and analyse

We want to know:
    * items with highest loss (i.e. most confidently wrong)
    * confusion matrix
    * movement of train and validation loss during training
"""
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd
import seaborn as sns  # type: ignore

from consts import get_consts

consts = get_consts()


class Analyser:
    def __init__(self, metrics: pd.DataFrame, outputs: pd.DataFrame) -> None:
        self.metrics = metrics
        self.outputs = outputs
        self.data = pd.read_pickle(consts["DATA"])

    def view_training_metrics(self, output_file: Optional[str] = None) -> None:
        plt.figure()
        for col in ("train_accuracy", "val_accuracy"):
            plot_df = self.metrics.dropna(subset=[col])
            plt.plot(plot_df["step"], plot_df[col], label=col)
        plt.legend()
        plt.title("Accuracy Values During Training")
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

    def view_confusion_matrix(self):
        cm = self.get_confusion_matrix()
        sns.heatmap(cm)
        plt.show()

    def view_image(self):
        pass

    def most_confidently_wrong(self, n: int = 20):
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
            is_valid = row["is_valid"]
            plt.title(
                "\n".join(
                    [
                        f"Label: {label}, Pred: {pred} ({100 * conf:.2f}%)",
                        "Validation set" if is_valid else "Training set",
                    ]
                )
            )
            plt.show()
        return most_confused


# TODO: method to run other functions to create an html report with the images
# TODO: this should be able to contain data from all folds somehow


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
    analyser.view_confusion_matrix()
    analyser.most_confidently_wrong()
