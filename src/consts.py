"""
Values in here should only change for a different environment or different
competition. If they change as part of testing or hyper-parameter tuning, they
belong in config.py
"""
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DATA_DIR = DATA_DIR / "output"

# Dataset properties
TEST_DATA_ROWS = 28000
TRAIN_DATA_ROWS = 42000
INPUT_SIZE = 784
N_CLASSES = 10
MINI_DATA_PROPORTION = 0.1
MICRO_DATA_PROPORTION = 0.01

# Random seed
SEED = 777
