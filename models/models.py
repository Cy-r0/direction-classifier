import torch.nn as nn



class SimpleCNN(nn.Module):
    """
    Simple 1D convolutional network.
    """

    def __init__(self, n_features):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 1, 3)

    def forward(self, x):
        x = self.conv1(x)
        return x
