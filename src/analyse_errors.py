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


class Analyser:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def view_training_loss(self, output_file: Optional[str] = None) -> None:
        plt.figure()
        for col in ("train_loss", "val_loss"):
            plot_df = self.df.dropna(subset=[col])
            plt.plot(plot_df["step"], plot_df[col], label=col)
        plt.legend()
        plt.title("Loss Values During Training")
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please provide a metrics file"
    metrics_file = Path(sys.argv[1])
    assert metrics_file.exists(), "The provided metrics file does not exist"

    analyser = Analyser(pd.read_csv(metrics_file))
    analyser.view_training_loss()
