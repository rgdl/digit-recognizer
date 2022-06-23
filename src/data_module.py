import pandas as pd  # type: ignore
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from consts import DATA


class DataModule(pl.LightningDataModule):
    def __init__(self, fold: int):
        super().__init__()
        self.fold = fold
        self.all_data = pd.read_pickle(DATA)

        assert 0 <= fold <= self.all_data["fold"].max()

        self._test_rows = self.all_data["fold"] < 0
        self._valid_rows = self.all_data["fold"] == fold
        self._train_rows = ~(self._test_rows | self._valid_rows)

    def train_dataloader(self):
        return DataLoader(self.all_data.loc[self._train_rows], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.all_data.loc[self._valid_rows], shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.all_data.loc[self._test_rows])
