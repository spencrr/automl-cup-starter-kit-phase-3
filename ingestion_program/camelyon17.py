from pathlib import Path

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader

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
        dataset = get_dataset(dataset='camelyon17', download=True, root_dir='/home/wilds_data/')
        if split == "train":
            train_dataset = dataset.get_subset("train")
            return train_dataset
            # train_dataloader = get_train_loader("standard", train_dataset, batch_size=batch_size)
        elif split == "test":
            test_dataset = dataset.get_subset("test")
            return test_dataset
            # test_dataloader = get_eval_loader("standard", test_dataset, batch_size=batch_size)