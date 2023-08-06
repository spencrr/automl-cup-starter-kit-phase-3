from pathlib import Path
import os


from tqdm import tqdm
import numpy as np

from dataloader import AutoMLCupDataloader

class Camelyon17Dataloader(AutoMLCupDataloader):
    @staticmethod
    def name():
        return "camelyon17"

    def __init__(self, directory: Path, **kwargs):
        super().__init__(directory, **kwargs)

        self.train = None
        self.test = None
    
    def get_split(self, split):
        if split == "train":
            if self.train is None:
                x_train = np.load(self.directory / "x_train.npz")
                y_train = np.load(self.directory / "y_train.npz")
                self.train = {
                    "input": x_train,
                    "label": y_train,
                }
            return self.train
            
            """
            x_files = np.sort([f_ for f_ in os.listdir(self.directory) if 'x_train' in f_])
            y_files = np.sort([f_ for f_ in os.listdir(self.directory) if 'y_train' in f_])
            x = []
            y = []
            print("loading camelyon....")
            for f_x, f_y in tqdm(zip(x_files, y_files)):
                print(f_x, f_y)
                x_ = np.load(os.path.join(self.directory, f_x))
                y_ = np.load(os.path.join(self.directory, f_y))
                x.append(x_)
                y.append(y_)
            x = np.concatenate(x, axis=0)
            y = np.concatenate(y, axis=0)
            self.train = {
                "input": x,
                "label": y,
            }
            return self.train
            """
            
        elif split == "test":
            if self.test is None:
                x_test = np.load(self.directory / "x_test.npz")
                y_test = np.load(self.directory / "y_test.npz")
                self.test = {
                    "input": x_test,
                    "label": y_test,
                }
            return self.test

            """
            x_files = np.sort([f_ for f_ in os.listdir(self.directory) if 'x_test' in f_])
            y_files = np.sort([f_ for f_ in os.listdir(self.directory) if 'y_test' in f_])
            x = []
            y = []
            print("loading camelyon....")
            for f_x, f_y in tqdm(zip(x_files, y_files)):
                print(f_x, f_y)
                x_ = np.load(os.path.join(self.directory, f_x))
                y_ = np.load(os.path.join(self.directory, f_y))
                x.append(x_)
                y.append(y_)
            x = np.concatenate(x, axis=0)
            y = np.concatenate(y, axis=0)
            self.test = {
                "input": x,
                "label": y,
            }
            return self.test
            """