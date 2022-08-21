from typing import Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

from consts import get_consts

consts = get_consts()
BatchType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class Dataset(BaseDataset):
    def __init__(self, data: pd.DataFrame) -> None:
        self.x = torch.FloatTensor(
            data[
                [
                    col
                    for col in data
                    if isinstance(col, str) and col.startswith("pixel")
                ]
            ].values
        )
        self.y = torch.LongTensor(data["label"].values)
        self.ind = torch.IntTensor(data.index)
        self._len = len(self.x)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> BatchType:
        return self.x[idx], self.y[idx], self.ind[idx]


class DataModule(pl.LightningDataModule):
    COMMON_DATA_LOADER_ARGS = {
        "batch_size": consts["BATCH_SIZE"],
        "num_workers": consts["N_WORKERS"],
        "drop_last": False,
    }

    def __init__(self, fold: int):
        super().__init__()
        self.fold = fold
        self.all_data = pd.read_pickle(consts["DATA"])

        assert 0 <= fold <= self.all_data["fold"].max()

        self._test_rows = self.all_data["fold"] < 0
        self._valid_rows = self.all_data["fold"] == fold
        self._train_rows = ~(self._test_rows | self._valid_rows)

    def train_dataloader(self):
        data = self.all_data.loc[self._train_rows].drop("fold", axis=1)
        ds = Dataset(data)
        return DataLoader(
            ds,
            **self.COMMON_DATA_LOADER_ARGS,  # type: ignore
            shuffle=True,
        )

    def val_dataloader(self):
        data = self.all_data.loc[self._valid_rows].drop("fold", axis=1)
        ds = Dataset(data)
        return DataLoader(ds, **self.COMMON_DATA_LOADER_ARGS)  # type: ignore

    def test_dataloader(self):
        data = self.all_data.loc[self._test_rows].drop("fold", axis=1)
        ds = Dataset(data)
        return DataLoader(ds, **self.COMMON_DATA_LOADER_ARGS)  # type: ignore
