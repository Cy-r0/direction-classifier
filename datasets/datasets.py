import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class DCDataset(Dataset):
    """
    Dataset class for Direction Classification (DC) of time series.
    Requires samples and target to be stored in separate directories named 
    "samples" and "targets".

    Args:
        root (string): root folder of dataset.
        transforms (callable): pytorch transforms to apply.
    """

    def __init__(self, root, transforms=None):
        self.root = root
        self.sample_dir = os.path.join(root, "samples")
        self.target_dir = os.path.join(root, "targets")
        self.transforms = transforms
        
        assert os.path.isdir(root), "Dataset root not found"

        self.samples = os.listdir(self.sample_dir)
        self.targets = os.listdir(self.target_dir)

        assert set(self.samples) == set(self.targets), \
            "Not all samples have a matching target"

    def __getitem__(self, index):
        sample_path = os.path.join(self.sample_dir, self.samples[index])
        target_path = os.path.join(self.target_dir, self.targets[index])

        sample = pd.read_csv(sample_path)
        target = pd.read_csv(target_path)

        datapoint = {"sample": sample, "target": target}

        if self.transforms is not None:
            datapoint = self.transforms(datapoint)

        return datapoint
    
    def __len__(self):
        return len(self.samples)



if __name__ == "__main__":
    dataset = DCDataset("USDFAK")
    print(dataset[0]["sample"])
    