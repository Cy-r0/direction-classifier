import torch


class Normalise(object):
    """
    Normalise features in sample between 0 and 1.
    """
    def __call__(self, datapoint):
        #TODO: implement
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