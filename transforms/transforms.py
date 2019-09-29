import torch


class Normalise(object):
    """
    Normalise features in sample between 0 and 1.
    """

    def __init__(self):
        pass

    def __call__(self, datapoint):
        #TODO: implement
        return datapoint


class ToTensor(object):
    """
    Convert both sample and target (pandas dataframes) to pytorch tensors.
    """

    def __call__(self, datapoint):
        sample = datapoint["sample"].to_numpy()
        target = datapoint["target"].to_numpy()
        print("from inside totensor", sample, target)
        sample = torch.Tensor(sample).transpose(-1, -2) # Transpose to BCL format
        target = torch.Tensor(target).transpose(-1, -2)
        print("after transposing tensor", sample, target)
        
        datapoint = {"sample": sample, "target": target}
        return datapoint