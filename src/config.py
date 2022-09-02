from typing import Any
from typing import Dict

from consts import IS_LOCAL  # script-gen: consts.py
from consts import PROCESSED_DATA_DIR  # script-gen: consts.py

_config = None


def get_config() -> Dict[str, Any]:
    """
    Wrapped into a function so elements can be over-ridden in tests, cached at
    the module level to get singleton behaviour
    """
    global _config
    if _config is None:
        _config = {
            # Training parameters
            "BATCH_SIZE": 32,
            "MAX_EPOCHS": 3,
            # Data prep
            "N_FOLDS": 5,
            # File paths
            "DATA": PROCESSED_DATA_DIR / "micro.pickle",
        }
        if IS_LOCAL:
            _config["N_WORKERS"] = 0
        else:
            _config["N_WORKERS"] = 2
    return _config
