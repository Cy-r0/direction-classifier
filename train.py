import argparse
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

def conf2fig(conf, fmt):
    """
    Convert confusion matrix to matplotlib figure.
    """
    fig = plt.figure(figsize=(5,4))
    sn.heatmap(
        pd.DataFrame(conf),
        annot=True,
        annot_kws={"size": 20},
        fmt=fmt,
        xticklabels=["0", "1"],
        yticklabels=["0", "1"])
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


def train(model, train_loader, val_loader, epochs, batch_size, lr, momentum, log):
    """
    Train model on training set.
    Args:
        model (torch.nn.Module): pytorch model to use.
        train_loader (DataLoader).
        val_loader (DataLoader).
        epochs (int): number of epochs to train for.
        batch_size (int).
        lr (float): optimiser learning rate.
        momentum (float): optimiser momentum.
        log (bool): whether to log on tensorboard or not.
    """
    # Fix random seeds
    np.random.seed(777)
    torch.manual_seed(777)

    # Setup gpu
    device = torch.device("cuda:0")
    print("Device:", device)
    
    model.to(device)

    # Initialise optimiser
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimiser = ConfigOptimiser(
        model.parameters(), 
        "SGD", 
        lr=lr, 
        momentum=momentum,
        dampening=0,
        weight_decay=0,
        nesterov=False).optimiser

    # Initialise tensorboard logger
    if log:
        logger = SummaryWriter(flush_secs=120)

        # Log model architecture (for some reason, a sample input is required)
        sample_batch = next(iter(train_loader))["sample"].to(device)
        logger.add_graph(model, sample_batch, verbose=False)

        # Log training hyperparameters
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
            sample = batch["sample"].to(device)
            # Squeeze because crossentropy wants 1d target
            target = batch["target"].long().squeeze().to(device) 

            optimiser.zero_grad()
            prediction = model(sample)

            argmax_prediction = torch.argmax(prediction, dim=1)
            loss = criterion(prediction, target)
            loss.backward()
            optimiser.step()

            # Accumulate loss and confusion
            train_loss += loss.item()
            train_conf += calc_conf(argmax_prediction.cpu(), target.cpu())
        
        # Normalise by number of batches
        train_loss /= len(train_loader)

        # -------------------
        # VAL LOOP
        # -------------------
        val_loader = tqdm(val_loader, desc="Valid", ascii=True)
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                sample = batch["sample"].to(device)
                target = batch["target"].long().squeeze().to(device)

                prediction = model(sample)

                argmax_prediction = torch.argmax(prediction, dim=1)
                loss = criterion(prediction, target)

                # Accumulate loss and confusion
                val_loss += loss.item()
                val_conf += calc_conf(argmax_prediction.cpu(), target.cpu())
        
        # Normalise by number of batches
        val_loss /= len(val_loader)

        if log:
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
    
    if log:
        # Convert confusion matrices to figures and log them
        train_conf_fig = conf2fig(train_conf, fmt="0.0f")
        val_conf_fig = conf2fig(val_conf, fmt="0.0f")
        logger.add_figure("train confusion", train_conf_fig, epoch)
        logger.add_figure("val confusion", val_conf_fig, epoch)

        # Normalise confusion matrices and log them again
        train_conf = normalise_conf(train_conf)
        val_conf = normalise_conf(val_conf)
        train_conf_fig = conf2fig(train_conf, fmt="0.3f")
        val_conf_fig = conf2fig(val_conf, fmt="0.3f")
        logger.add_figure("train confusion normalised", train_conf_fig, epoch)
        logger.add_figure("val confusion normalised", val_conf_fig, epoch)

        logger.close()


if __name__ == "__main__":

    # Parse args
    parser = argparse.ArgumentParser(description="Train network.")
    parser.add_argument("--log", type=str, default="n", help="Log on tensorboard?")
    args = parser.parse_args()
    logstr = args.log
    if logstr == "Y" or logstr == "y":
        log = True
    elif logstr == "N" or logstr == "n":
        log = False
    else:
        raise ValueError("Command line argument invalid. Choose between y/n.")

    # Init transforms
    in_feat = ["MonthDay", "WeekDay", "Hour", "Open", "High", "Low", "Close"]
    out_feat = ["Target"]
    mean = pd.DataFrame(
        [[16., 3., 11.5, "auto", "auto", "auto", "auto"]],
        columns=in_feat)
    std = pd.DataFrame(
        [[8.944272, 1.414214, 6.922187, "auto", "auto", "auto", "auto"]],
        columns=in_feat)
    transforms = T.Compose([
        myT.SelectFeatures(
            input_features=in_feat,
            output_features=out_feat),
        myT.Normalise(
            mean=mean,
            std=std),
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
        epochs=50, 
        batch_size=batch_size,
        lr=0.001, 
        momentum=0.0,
        log=False)
