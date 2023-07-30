from pathlib import Path
from typing import Dict

import numpy as np
from numpy.typing import ArrayLike

from dataloader import AutoMLCupDataloader


class GlobalWheatDataloader(AutoMLCupDataloader):
    @staticmethod
    def name():
        return "globalwheat"

    def __init__(self, directory: Path, **kwargs):
        super().__init__(directory, **kwargs)

        self.train: Dict[str, ArrayLike] | None = None
        self.test: Dict[str, ArrayLike] | None = None

    def get_split(self, split):
        if split == "train":
            if self.train is None:
                x_train = np.load(self.directory / "x_train.npy")
                y_train = np.load(self.directory / "y_train.npy")
                self.train = {
                    "input": x_train,
                    "label": y_train,
                }

            return self.train
        if split == "test":
            if self.test is None:
                x_test = np.load(self.directory / "x_test.npy")
                y_test = np.load(self.directory / "y_test.npy")
                self.test = {
                    "input": x_test,
                    "label": y_test,
                }

            return self.test