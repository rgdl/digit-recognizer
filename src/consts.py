"""
Values in here should only change for a different environment or different
competition. If they change as part of testing or hyper-parameter tuning, they
belong in config.py
"""
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
IS_LOCAL = (ROOT_DIR / ".local").exists()
if IS_LOCAL:
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    OUTPUT_DATA_DIR = DATA_DIR / "output"
    DEVICES = None
    ACCELERATOR = None
else:
    DATA_DIR = ROOT_DIR
    RAW_DATA_DIR = DATA_DIR / "input/digit-recognizer"
    PROCESSED_DATA_DIR = DATA_DIR
    OUTPUT_DATA_DIR = DATA_DIR
    DEVICES = 1
    ACCELERATOR = "gpu"

# Dataset properties
TEST_DATA_ROWS = 28000
TRAIN_DATA_ROWS = 42000
INPUT_SIZE = 784
N_CLASSES = 10
MINI_DATA_PROPORTION = 0.1
MICRO_DATA_PROPORTION = 0.01

# Random seed
SEED = 777
