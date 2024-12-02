import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, latent_dim=None):
        super().__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(
                nn.Conv2d(in_channels, out_channels=h_dim,
                          kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(h_dim))
            layers.append(nn.LeakyReLU())
            in_channels = h_dim
        if latent_dim is not None:
            layers.append(nn.Flatten())
            layers.append(nn.Linear(hidden_dims[-1]*4, latent_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class LinearEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim=None):
        super().__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU())
            input_dim = h_dim
        # Final layer maps to latent dimension
        if latent_dim is not None:
            layers.append(nn.Linear(input_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)




