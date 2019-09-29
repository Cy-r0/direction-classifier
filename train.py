import numpy as np
import torch
from tqdm import tqdm

from setup import ConfigDataset, ConfigModel
import transforms as myT

def train(symbol, epochs, batch_size, lr, momentum):
    """
    Train model on training set.
    Args:
        symbol (string): name of commodity.
        epochs (int): number of epochs to train for.
        batch_size (int).

    """

    transforms = myT.ToTensor()

    c = ConfigDataset(
        symbol,
        batch_size=batch_size, 
        train_idx=[0,1], 
        val_idx=[2], 
        test_idx=[3], 
        transforms=transforms)
    train_loader = c.train_loader
    val_loader = c.val_loader

    model = ConfigModel("SimpleCNN").model

    criterion = torch.nn.BCEWithLogitsLoss()
    optimiser = ConfigOptimiser("SGD", lr=lr, momentum=momentum).optimiser
    

    for epoch in range(epochs):

        train_loader = tqdm(train_loader, desc="Train", ascii=True)

        for datapoint in train_loader:
            sample = datapoint["sample"]
            target = datapoint["target"]

            prediction = model(sample)

            loss = 

            pass

        val_loader = tqdm(val_loader, desc="Valid", ascii=True)

        for datapoint in val_loader:
            sample = datapoint["sample"]
            target = datapoint["target"]





if __name__ == "__main__":
    train("USDFAK", 500, 32)