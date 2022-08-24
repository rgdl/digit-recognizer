from pathlib import Path
from typing import Any
from typing import Dict

_consts = None


def get_consts() -> Dict[str, Any]:
    """
    Wrapped into a function so elements can be over-ridden in tests, cached at
    the module level to get singleton behaviour
    """
    global _consts
    if _consts is None:
        ROOT_DIR = Path(__file__).parent.parent
        DATA_DIR = ROOT_DIR / "data"
        RAW_DATA_DIR = DATA_DIR / "raw"
        PROCESSED_DATA_DIR = DATA_DIR / "processed"
        OUTPUT_DATA_DIR = DATA_DIR / "output"

        DATA = PROCESSED_DATA_DIR / "mini.pickle"

        assert DATA.exists()

        _consts = {
            # Dataset properties
            "TEST_DATA_ROWS": 28000,
            "TRAIN_DATA_ROWS": 42000,
            "INPUT_SIZE": 784,
            "N_CLASSES": 10,
            # Training parameters
            "N_WORKERS": 4,
            "BATCH_SIZE": 32,
            # Data prep
            "N_FOLDS": 5,
            "MINI_DATA_PROPORTION": 0.1,
            "MICRO_DATA_PROPORTION": 0.01,
            # File paths
            "ROOT_DIR": ROOT_DIR,
            "DATA_DIR": DATA_DIR,
            "RAW_DATA_DIR": RAW_DATA_DIR,
            "PROCESSED_DATA_DIR": PROCESSED_DATA_DIR,
            "OUTPUT_DATA_DIR": OUTPUT_DATA_DIR,
            "DATA": DATA,
            # Random seed
            "SEED": 777,
        }
    return _consts
