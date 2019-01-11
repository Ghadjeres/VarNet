import torch

from torch import nn


class Discriminator(nn.Module):
    def __init__(self, z_dim, num_style_tokens, hidden_size=128, dropout_prob=0.5):
        super(Discriminator, self).__init__()
        self.dropout_prob = dropout_prob

        self.disc = nn.Sequential(
            nn.Linear(in_features=z_dim + num_style_tokens, out_features=hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(in_features=hidden_size, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, z, alpha):
        z_and_alpha = torch.cat([z, alpha], 1)
        return self.disc.forward(z_and_alpha)
