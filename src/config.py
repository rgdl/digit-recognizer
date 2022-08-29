from typing import Any
from typing import Dict

from consts import PROCESSED_DATA_DIR

_config = None


def get_config() -> Dict[str, Any]:
    """
    Wrapped into a function so elements can be over-ridden in tests, cached at
    the module level to get singleton behaviour
    """
    global _config
    if _config is None:

        DATA = PROCESSED_DATA_DIR / "micro.pickle"

        assert DATA.exists()

        _config = {
            # Training parameters
            "N_WORKERS": 4,
            "BATCH_SIZE": 32,
            "MAX_EPOCHS": 3,
            # Data prep
            "N_FOLDS": 5,
            # File paths
            "DATA": DATA,
        }
    return _config
