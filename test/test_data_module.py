import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from consts import BATCH_SIZE
from consts import INPUT_SIZE
from consts import N_FOLDS
from consts import TEST_DATA_ROWS
from consts import TRAIN_DATA_ROWS
from data_module import DataModule


def test_data_module():
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


def test_batching():
    dm = DataModule(fold=0)

    x_train, y_train = next(iter(dm.train_dataloader()))
    assert x_train.shape == (BATCH_SIZE, INPUT_SIZE)
    assert y_train.shape == (BATCH_SIZE,)
    assert all((0 <= y_train) & (y_train <= 9))

    x_val, y_val = next(iter(dm.val_dataloader()))
    assert x_val.shape == (BATCH_SIZE, INPUT_SIZE)
    assert y_val.shape == (BATCH_SIZE,)
    assert all((0 <= y_val) & (y_val <= 9))

    x_test, y_test = next(iter(dm.test_dataloader()))
    assert x_test.shape == (BATCH_SIZE, INPUT_SIZE)
    assert y_test.shape == (BATCH_SIZE,)
    assert all(y_test == -1)
