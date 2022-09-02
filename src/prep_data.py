from pathlib import Path

import pandas as pd

from config import get_config  # script-gen: config.py
from consts import MICRO_DATA_PROPORTION  # script-gen: consts.py
from consts import MINI_DATA_PROPORTION  # script-gen: consts.py
from consts import PROCESSED_DATA_DIR  # script-gen: consts.py
from consts import RAW_DATA_DIR  # script-gen: consts.py
from consts import SEED  # script-gen: consts.py

config = get_config()


def main(output_dir: Path) -> None:
    train_df = pd.read_csv(RAW_DATA_DIR / "train.csv").sort_values("label")
    train_df["fold"] = [i % config["N_FOLDS"] for i in range(len(train_df))]
    train_df = train_df.sort_index()

    test_df = pd.read_csv(RAW_DATA_DIR / "test.csv").assign(label=-1, fold=-1)

    df = pd.concat([train_df, test_df])

    df.to_pickle(str(output_dir / "full.pickle"))
    df.groupby(["label", "fold"]).sample(
        frac=MINI_DATA_PROPORTION, random_state=SEED
    ).to_pickle(str(output_dir / "mini.pickle"))
    df.groupby(["label", "fold"]).sample(
        frac=MICRO_DATA_PROPORTION, random_state=SEED
    ).to_pickle(str(output_dir / "micro.pickle"))


if __name__ == "__main__":
    main(PROCESSED_DATA_DIR)
