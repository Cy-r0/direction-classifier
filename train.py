import warnings

import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from tqdm import tqdm

from config import ConfigDataset, ConfigModel, ConfigOptimiser
import transforms as myT



def calc_conf(predictions, targets):
    """
    Calculate confusion matrix. Requires predictions to be argmaxed.
    """
    predictions = predictions.flatten()
    targets = targets.flatten()
    conf = confusion_matrix(targets, predictions)
    return conf


def normalise_conf(conf):
    """
    Normalise confusion matrix.
    """
    conf_n = np.zeros(conf.shape)

    for row in range(len(conf)):
        # Avoid division by zero
        if conf[row].sum() != 0:
            conf_n[row] = conf[row] / conf[row].sum()
        else:
            warnings.warn("Row of confusion matrix is zero")
            conf_n[row] = conf[row]
    
    return conf_n


def train(symbol, input_f, output_f, epochs, batch_size, lr, momentum):
    """
    Train model on training set.
    Args:
        symbol (string): name of commodity.
        input_f (sequence): list of input features.
        output_f (sequence): list of output features.
        epochs (int): number of epochs to train for.
        batch_size (int).
        lr (float): optimiser learning rate.
        momentum (float): optimiser momentum.
    """

    transforms = T.Compose([
        myT.SelectFeatures(
            input_features=input_f,
            output_features=output_f),
        myT.ToTensor()
    ])

    # TODO write some code to generate train and val indices

    # Initialise dataset, model, optimiser and tensorboard logger
    c = ConfigDataset(
        symbol,
        batch_size=batch_size, 
        train_idx=[0,1], 
        val_idx=[2], 
        test_idx=[3], 
        transforms=transforms)
    train_loader = c.train_loader
    val_loader = c.val_loader

    model = ConfigModel(
        "SimpleCNN",
        input_features=len(input_f), 
        output_classes=len(output_f)).model

    criterion = torch.nn.CrossEntropyLoss()
    optimiser = ConfigOptimiser(
        model.parameters(), 
        "SGD", 
        lr=lr, 
        momentum=momentum,
        dampening=0,
        weight_decay=0,
        nesterov=False).optimiser

    logger = SummaryWriter(flush_secs=120)

    for epoch in range(epochs):
        # Initialise all things to keep track of 
        train_loss = 0.
        train_conf = np.zeros((2,2), dtype="float")
        val_loss = 0.
        val_conf = np.zeros((2,2), dtype="float")

        # ---------------------
        # TRAIN LOOP
        # ---------------------
        train_loader = tqdm(train_loader, desc="Train", ascii=True)
        model.train()
        for batch in train_loader:
            sample = batch["sample"]
            target = batch["target"].long().squeeze(1)

            optimiser.zero_grad()
            prediction = model(sample).squeeze()

            argmax_prediction = torch.argmax(prediction, dim=1)
            loss = criterion(prediction, target)
            loss.backward()
            optimiser.step()

            # Accumulate loss and confusion
            train_loss += loss.item()
            train_conf += calc_conf(argmax_prediction, target)
        
        # Normalise by number of batches
        train_loss /= len(train_loader)
        train_conf /= len(train_loader)

        # -------------------
        # VAL LOOP
        # -------------------
        val_loader = tqdm(val_loader, desc="Valid", ascii=True)
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                sample = batch["sample"]
                target = batch["target"]

                pass

        # Log on tensorboard
        logger.add_scalars("losses", {"train": train_loss, "val": val_loss},
            epoch)
        train_acc = np.trace(train_conf) / np.sum(train_conf)
        val_acc = np.trace(val_conf) / np.sum(val_conf)
        logger.add_scalars("acc", {"train": train_acc, "val": val_acc},
            epoch)
        logger.add_figure("confusion", {"train": train_conf, "val": val_conf},
            epoch)

        # Normalise confusion matrices and log again
        train_conf = normalise_conf(train_conf)
        val_conf = normalise_conf(val_conf)
        logger.add_figure("confusion normalised",
            {"train": train_conf, "val": val_conf},
            epoch)



if __name__ == "__main__":
    train(
        "USDFAK",
        input_f=["Hour", "Open", "High", "Low", "Close"], 
        output_f=["Target"], 
        epochs=5, 
        batch_size=2, 
        lr=0.01, 
        momentum=0.9)