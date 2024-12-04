import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, input_channels=3, hidden_dims=[32,64,128]):
        super(ConvEncoder, self).__init__()

        self.hidden_dims = hidden_dims

        modules = []
        in_channels = input_channels
        # Build Encoder Layers
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        x = self.encoder(x)
        return x


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




