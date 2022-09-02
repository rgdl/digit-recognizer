from typing import Any
from typing import Dict
from typing import List

import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only


class Logger(pl.loggers.LightningLoggerBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metrics: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "Logger"

    @property
    def version(self) -> str:
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params) -> None:
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step) -> None:
        self.metrics.append({"step": step, **metrics})
