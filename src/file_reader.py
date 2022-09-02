from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Union

import pandas as pd

_file_data: Dict[str, Any] = {}
_read_counts: Dict[str, int] = {}
VERBOSE = True


def read_csv(file: Union[str, Path]) -> pd.DataFrame:
    return _read_file(file, pd.read_csv)


def read_pickle(file: Union[str, Path]) -> pd.DataFrame:
    return _read_file(file, pd.read_pickle)


def _read_file(file: Union[str, Path], read_func: Callable) -> pd.DataFrame:
    global _file_data
    key = str(Path(file).absolute())
    if key not in _file_data:
        _file_data[key] = read_func(file)
        _read_counts[key] = 0
    _read_counts[key] += 1
    if VERBOSE:
        print(_read_counts)
    return _file_data[key]
