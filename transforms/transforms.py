import numpy as np
import pandas as pd
import torch


class Normalise(object):
    """
    Normalise features in sample according to given mean and std.
    Args:
        mean (pandas DataFrame).
        std (pandas DataFrame).
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
        # Find columns that dont contain "auto" values in mean and std
        self.cols_mean = mean.loc[:, mean.loc[0, :]!="auto"].columns
        self.cols_std = std.loc[:, std.loc[0, :]!="auto"].columns
        # Also find indices of these columns
        self.cols_mean_i = [self.mean.columns.get_loc(col) for col in self.cols_mean]
        self.cols_std_i = [self.std.columns.get_loc(col) for col in self.cols_std]
        
        # Convert dataframes to Series
        self.mean = self.mean.iloc[0, :]
        self.std = self.std.iloc[0, :]


    def __call__(self, datapoint):
        sample = datapoint["sample"]
        target = datapoint["target"]

        # Reorder columns in stats to match order of sample
        # Will throw an error if stats dont contain all the headers in sample
        #self.mean = self.mean[sample.columns]
        #self.std = self.std[sample.columns]

        # Calculate current mean and std of the sample
        automean = np.mean(sample.values, axis=0)
        autostd = np.mean(sample.values, axis=0)
        
        # Replace values that dont correspond to "auto" with values from self stats
        automean[self.cols_mean_i] = self.mean[self.cols_mean]
        autostd[self.cols_std_i] = self.std[self.cols_std]

        # Apply normalisation to sample
        sample = (sample - automean) / autostd

        datapoint = {"sample": sample, "target": target}
        return datapoint
        

class SelectFeatures(object):
    """
    Choose which features to keep in sample and target (pandas dataframes).
    Args:
        input_features (sequence): list of input features to keep.
        output_features (sequence): list of output features to keep.
    """
    def __init__(self, input_features, output_features):
        self.input_features = input_features
        self.output_features = output_features

    def __call__(self, datapoint):
        sample = datapoint["sample"]
        target = datapoint["target"]

        sample = sample[self.input_features]
        target = target[self.output_features]

        datapoint = {"sample": sample, "target": target}
        return datapoint


class ToTensor(object):
    """
    Convert both sample and target (pandas dataframes) to pytorch tensors.
    """
    def __call__(self, datapoint):
        sample = datapoint["sample"].to_numpy()
        target = datapoint["target"].to_numpy()
        sample = torch.Tensor(sample).transpose(-1, -2) # Transpose to BCL format
        target = torch.Tensor(target).transpose(-1, -2)
        
        datapoint = {"sample": sample, "target": target}
        return datapoint