import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
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

def conf2fig(conf):
    """
    Convert confusion matrix to matplotlib figure.
    """
    fig = plt.figure(figsize=(5,4))
    sn.heatmap(
        pd.DataFrame(conf),
        annot=True)
    plt.ylabel("True")
    plt.xlabel("Predicted")

    return fig

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


def train(model, train_loader, val_loader, epochs, batch_size, lr, momentum):
    """
    Train model on training set.
    Args:
        model (nn.Module): pytorch model to use.
        train_loader (DataLoader).
        val_loader (DataLoader).
        epochs (int): number of epochs to train for.
        batch_size (int).
        lr (float): optimiser learning rate.
        momentum (float): optimiser momentum.
    """

    # Initialise optimiser
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
    # TODO log network hyperparams as well
    hyperparams = "batch_size: %d, lr: %f, momentum: %f" \
        %(batch_size, lr, momentum)
    logger.add_text("training hyperparameters", hyperparams)

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
            target = batch["target"].long().squeeze() # Crossentropy wants 1d target

            optimiser.zero_grad()
            prediction = model(sample)

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
                target = batch["target"].long().squeeze()

                prediction = model(sample)

                argmax_prediction = torch.argmax(prediction, dim=1)
                loss = criterion(prediction, target)

                # Accumulate loss and confusion
                val_loss += loss.item()
                val_conf += calc_conf(argmax_prediction, target)
        
        # Normalise by number of batches
        val_loss /= len(val_loader)
        val_conf /= len(val_loader)

        # Log scalars on tensorboard
        logger.add_scalars("losses", {"train": train_loss, "val": val_loss},
            epoch)
        train_acc = np.trace(train_conf) / np.sum(train_conf)
        val_acc = np.trace(val_conf) / np.sum(val_conf)
        logger.add_scalars("acc", {"train": train_acc, "val": val_acc},
            epoch)
        
        # Print some info
        print("Epoch: %d/%d. Train loss: %f, val loss: %f. " \
            "Train acc: %f, val acc: %f."
            %(epoch + 1, epochs, train_loss, val_loss, train_acc, val_acc))
    
    # Convert confusion matrices to figures and log them
    train_conf_fig = conf2fig(train_conf)
    val_conf_fig = conf2fig(val_conf)
    logger.add_figure("train confusion", train_conf_fig, epoch)
    logger.add_figure("val confusion", val_conf_fig, epoch)

    # Normalise confusion matrices and log them again
    train_conf = normalise_conf(train_conf)
    val_conf = normalise_conf(val_conf)
    train_conf_fig = conf2fig(train_conf)
    val_conf_fig = conf2fig(val_conf)
    logger.add_figure("train confusion normalised", train_conf, epoch)
    logger.add_figure("val confusion normalised", val_conf, epoch)

    logger.close()


if __name__ == "__main__":

    # Init transforms
    in_feat = ["Hour", "Open", "High", "Low", "Close"]
    out_feat = ["Target"]
    transforms = T.Compose([
        myT.SelectFeatures(
            input_features=in_feat,
            output_features=out_feat),
        myT.ToTensor()
    ])

    # Setup dataset
    batch_size = 32
    c = ConfigDataset(
        "EURUSD",
        batch_size=batch_size, 
        transforms=transforms)
    dataset_len = len(c.dataset)

    # Split dataset into train and validation
    train_split = int(0.5 * dataset_len)
    val_split = int(0.75 * dataset_len)
    train_idx = list(range(0, train_split))
    val_idx = list(range(train_split, val_split))
    c.split(train_idx=train_idx, val_idx=val_idx, test_idx=None)
    train_loader = c.train_loader
    val_loader = c.val_loader

    # Config model
    model = ConfigModel(
        "MLP",
        input_features=len(in_feat) * 240, 
        output_classes=2).model

    # Train model
    train(
        model,
        train_loader,
        val_loader,
        epochs=3, 
        batch_size=batch_size,
        lr=0.01, 
        momentum=0.9)