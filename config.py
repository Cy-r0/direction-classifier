import os

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T

from datasets import ForexDataset
import models
import transforms as myT


class ConfigDataset(object):
    """
    Configures a dataset, splitting it into train and test sets.
    Args:
        symbol (string): name of commodity.
        batch_size (int).
        n_workers (int): number of workers for dataloaders.
        transforms (callable): transforms to apply to each datapoint.
        train_idx (sequence): indices of dataset samples to be used for training.
        val_idx (sequence): indices for validation.
        test_idx (sequence): indices for testing.
    """

    def __init__(self,
        symbol,
        batch_size,
        n_workers=8,
        transforms=None,
        train_idx=None,
        val_idx=None,
        test_idx=None,):

        self.symbol = symbol
        self.batch_size = batch_size

        root = os.path.join("datasets", "prepared_data", symbol)
        assert os.path.exists(root), \
            "dataset root not found, are you sure it exists for this symbol?"
        self.dataset = ForexDataset(root, transforms=transforms)

        if train_idx:
            self.train_set = Subset(self.dataset, train_idx)
            self.train_loader = DataLoader(
                self.train_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=n_workers,
                pin_memory=True,
                drop_last=True)

        if val_idx:
            self.val_set = Subset(self.dataset, val_idx)
            self.val_loader = DataLoader(
                self.val_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=n_workers,
                pin_memory=True,
                drop_last=True)

        if test_idx:
            self.test_set = Subset(self.dataset, test_idx)
            self.test_loader = DataLoader(
                self.test_set,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                drop_last=False)


class ConfigModel(object):
    """
    Class that configures model, choosing from a model zoo.
    Args:
        model (string): name of model to use.
    """

    def __init__(self, model):
        if model == "SimpleCNN":
            self.model = models.SimpleCNN
            


if __name__ == "__main__":
    t = T.Compose([
        myT.ToTensor(),
    ])
    c = ConfigDataset(
        "USDFAK",
        batch_size=1, 
        train_idx=[0,1], 
        val_idx=[2], 
        test_idx=[3], 
        transforms=t)

    for datapoint in c.test_loader:
        print(datapoint)
    print(len(c.test_loader))

    model = ConfigModel("SimpleCNN").model

