#%%
from typing import List, Callable, Union, Any, TypeVar, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from models.base_vae import BaseVAE
from models.encoders import ConvEncoder
from models.decoders import ConvDecoder

class VanillaVAE(nn.Module): # BaseVAE):
    def __init__(self,
                latent_dim: int,
                in_channels: int = 3,
                hidden_dims: List = None,
                width: int = 32,
                height: int = 32,
                **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims

        self.encoder = ConvEncoder(in_channels, hidden_dims)

        # Compute the shape of the encoder output
        with torch.no_grad():
            self._encoder_shape = self.encoder(torch.zeros(1, in_channels, width, height)).shape

        # Compute the shape of tensor if we want to get the decoder output same as the input image
        

        self.fc_mu = nn.Linear(self._encoder_shape.numel(), latent_dim)
        self.fc_var = nn.Linear(self._encoder_shape.numel(), latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self._encoder_shape.numel())
        self.decoder = ConvDecoder(hidden_dims[::-1], in_channels)
        self.decoder_final = nn.Sequential(*[nn.Sigmoid()])


    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_input(z)
        x = x.view(-1, *self._encoder_shape[1:])
        x = self.decoder(x)
        x = self.decoder_final(x)
        return x

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return self.decode(z), x, mu, log_var

#%%
if __name__ == "__main__":
    vae_config = {
        'in_channels': 1,
        'latent_dim': 2,
        'hidden_dims': [32, 64, ],
        'width': 28, 'height': 28}

    vae_model = VanillaVAE(**vae_config).eval()
    dummy_input = torch.randn(7, *(vae_config[k] for k in['in_channels', 'width', 'height']))

    mu, log_var = vae_model.encode(dummy_input)
    z = vae_model.reparameterize(mu, log_var)
    print("z.shape=", z.shape)
    print("z=", z)

    decoder_input = vae_model.decoder_input(z)
    print("decoder_input.shape=", decoder_input.shape)

    x_hat = vae_model.decode(z)
    print("x_hat.shape=", x_hat.shape)
    # # Perform a forward pass
    # output = vae_model(dummy_input)
    # print("output[0].shape=", output[0].shape)
    # print("output[1].shape=", output[1].shape)
    # print("output[2].shape=", output[2].shape) 
    # print("output[3].shape=", output[3].shape)   # Check reconstruction and input shapes
# %%
