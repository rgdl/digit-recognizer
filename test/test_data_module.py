import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from consts import N_FOLDS
from consts import TEST_DATA_ROWS
from consts import TRAIN_DATA_ROWS
from data_module import DataModule


def test_data_module():
    """
    TEST_DATA_ROWS = 28000
    TRAIN_DATA_ROWS = 42000
    N_FOLDS = 5
    """

    dm = DataModule(fold=0)
    train = dm.train_dataloader()
    valid = dm.val_dataloader()
    test = dm.test_dataloader()

    # High tolerance needed for micro dataset
    tol = 3e-2

    expected_train_valid_ratio = 1 / (N_FOLDS - 1)
    actual_train_valid_ratio = len(valid) / len(train)
    assert abs(expected_train_valid_ratio - actual_train_valid_ratio) < tol

    expected_test_to_non_test_ratio = TEST_DATA_ROWS / TRAIN_DATA_ROWS
    actual_test_to_non_test_ratio = len(test) / (len(train) + len(valid))
    assert (
        abs(expected_test_to_non_test_ratio - actual_test_to_non_test_ratio)
        < tol
    )
