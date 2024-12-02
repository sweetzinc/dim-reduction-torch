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


        self.fc_mu = nn.Linear(self._encoder_shape.numel(), latent_dim)

#%%
if __name__ == "__main__":
    vae_config = {
        'in_channels': 1,
        'latent_dim': 2,
        'hidden_dims': [32, 64, ],
        'width': 28, 'height': 28}

    vae_model = VanillaVAE(**vae_config)
    dummy_input = torch.randn(4, *(vae_config[k] for k in['in_channels', 'width', 'height']))

    # Perform a forward pass
    output = vae_model(dummy_input)
    print("output[0].shape=", output[0].shape)
    print("output[1].shape=", output[1].shape)
    print("output[2].shape=", output[2].shape) 
    print("output[3].shape=", output[3].shape)   # Check reconstruction and input shapes