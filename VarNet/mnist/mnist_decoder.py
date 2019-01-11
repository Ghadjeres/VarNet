import torch
from torch import nn


class MNISTDecoder(nn.Module):
    def __init__(self, z_dim, hidden_size=400):
        super(MNISTDecoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, hidden_size)
        self.fc21 = nn.Linear(hidden_size, 784)

    def forward(self, z, **kwargs):
        """

        :param z:
        :param kwargs:
        :return: tuple forward pass and samples
        """
        h1 = torch.relu(self.fc1(z))
        x_reconstruct = torch.sigmoid(self.fc21(h1))
        return x_reconstruct, x_reconstruct
