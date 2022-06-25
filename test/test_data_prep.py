import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict

import pandas as pd  # type: ignore
import pytest

sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.consts import MICRO_DATA_PROPORTION  # type: ignore
from src.consts import MINI_DATA_PROPORTION  # type: ignore
from src.consts import N_FOLDS  # type: ignore
from src.prep_data import main  # type: ignore


@pytest.fixture(scope="session")
def datasets() -> Dict[str, pd.DataFrame]:
    with TemporaryDirectory() as _td:
        td = Path(_td)
        main(output_dir=td)
        return {
            name: pd.read_pickle(td / f"{name}.pickle")
            for name in ("full", "mini", "micro")
        }


def test_datasets_contains_all_folds(datasets):
    for df in datasets.values():
        assert {-1, *range(N_FOLDS)} == set(df["fold"])


def test_datasets_have_correct_assignment_to_test_fold(datasets):
    for df in datasets.values():
        assert all((df["label"] == -1) == (df["fold"] == -1))


def test_distribution_of_folds_and_labels(datasets):
    for x, df in datasets.items():
        counts = (
            df.loc[df["fold"] >= 0]
            .assign(count=1)
            .groupby(["label", "fold"], as_index=False)["count"]
            .sum()
            .pivot("label", "fold", "count")
        )
        smallest_folds = counts.min(axis=1)
        biggest_folds = counts.max(axis=1)
        assert max(biggest_folds - smallest_folds) <= 1


def test_correct_label_values(datasets):
    for df in datasets.values():
        train = df.loc[df["fold"] >= 0]
        test = df.loc[df["fold"] < 0]
        assert set(train["label"]) == set(range(10))
        assert set(test["label"]) == {-1}


def test_datasets_have_expected_relative_sizes(datasets):
    expected_mini = int(len(datasets["full"]) * MINI_DATA_PROPORTION)
    expected_micro = int(len(datasets["full"]) * MICRO_DATA_PROPORTION)
    assert abs(1 - expected_mini / len(datasets["mini"])) < 0.01
    assert abs(1 - expected_micro / len(datasets["micro"])) < 0.01
