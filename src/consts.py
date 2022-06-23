from pathlib import Path

TEST_DATA_ROWS = 28000
TRAIN_DATA_ROWS = 42000
N_FOLDS = 5

MINI_DATA_PROPORTION = 0.1
MICRO_DATA_PROPORTION = 0.01

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DATA = PROCESSED_DATA_DIR / "micro.pickle"
assert DATA.exists()

SEED = 777
