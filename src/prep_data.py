from pathlib import Path

import pandas as pd

from consts import get_consts

consts = get_consts()

MICRO_DATA_PROPORTION = consts["MICRO_DATA_PROPORTION"]
MINI_DATA_PROPORTION = consts["MINI_DATA_PROPORTION"]
N_FOLDS = consts["N_FOLDS"]
PROCESSED_DATA_DIR = consts["PROCESSED_DATA_DIR"]
RAW_DATA_DIR = consts["RAW_DATA_DIR"]
SEED = consts["SEED"]


def main(output_dir: Path) -> None:
    train_df = pd.read_csv(consts["RAW_DATA_DIR"] / "train.csv").sort_values(
        "label"
    )
    train_df["fold"] = [i % consts["N_FOLDS"] for i in range(len(train_df))]
    train_df = train_df.sort_index()

    test_df = pd.read_csv(consts["RAW_DATA_DIR"] / "test.csv").assign(
        label=-1, fold=-1
    )

    df = pd.concat([train_df, test_df])

    df.to_pickle(str(output_dir / "full.pickle"))
    df.groupby(["label", "fold"]).sample(
        frac=consts["MINI_DATA_PROPORTION"], random_state=consts["SEED"]
    ).to_pickle(str(output_dir / "mini.pickle"))
    df.groupby(["label", "fold"]).sample(
        frac=consts["MICRO_DATA_PROPORTION"], random_state=consts["SEED"]
    ).to_pickle(str(output_dir / "micro.pickle"))


if __name__ == "__main__":
    main(consts["PROCESSED_DATA_DIR"])
