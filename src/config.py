from typing import Any
from typing import Dict

from consts import IS_LOCAL  # script-gen: consts.py
from consts import PROCESSED_DATA_DIR  # script-gen: consts.py

_config = None
ConfigType = Dict[str, Any]


def get_config() -> ConfigType:
    """
    Wrapped into a function so elements can be over-ridden in tests, cached at
    the module level to get singleton behaviour
    """
    global _config
    if _config is None:
        _config = {
            # Architectural parameters
            "ARCHITECTURE": "BasicLinearModel",
            # Training parameters
            "BATCH_SIZE": 64,
            "MAX_EPOCHS": 16,
            "OPTIM_CLASS": "SGD",
            "OPTIM_PARAMS": {"lr": 3e-3},
            "LOSS_FUNC_CLASS": "CrossEntropyLoss",
            "LOSS_FUNC_PARAMS": {},
            # Data prep
            "N_FOLDS": 5,
            # File paths
            "DATA": PROCESSED_DATA_DIR / "micro.pickle",
        }
        if IS_LOCAL:
            _config["N_WORKERS"] = 0
        else:
            _config["N_WORKERS"] = 2
            _config["DATA"] = PROCESSED_DATA_DIR / "full.pickle"
    return _config
