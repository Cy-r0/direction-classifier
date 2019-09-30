import torch.nn as nn



class SimpleCNN(nn.Module):
    """
    Simple 1D convolutional network.
    Args:
        input_features (int): number of input features.
        output_classes (int): number of output classes.
    """

    def __init__(self, input_features, output_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(input_features, output_classes, 1)

    def forward(self, x):
        x = self.conv1(x)
        return x
