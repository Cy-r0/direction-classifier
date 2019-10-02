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

    def __call__(self, datapoint):
        sample = datapoint["sample"]
        target = datapoint["target"]

        # Reorder columns in stats to match order of sample
        # Will throw an error if stats dont contain all the headers in sample
        self.mean = self.mean[sample.columns]
        self.std = self.std[sample.columns]

        print(sample)

        for i in range(len(mean.columns)):

            # Calculate mean on the fly
            if mean.iloc[:, i] == "auto":
                automean = sample.iloc[:, i].mean()
                sample.iloc[:, i] -= automean
            # Use given mean
            else:
                sample.iloc[:, i] -= self.mean.at[0, i]

            # Calculate std on the fly
            if std.iloc[:, i] == "auto":
                autostd = sample.iloc[:, i].std()
                sample.iloc[:, i] /= autostd
            # Use given std
            else:
                sample.iloc[:, i] /= self.std.at[0, i]
        
        print(sample)
                


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