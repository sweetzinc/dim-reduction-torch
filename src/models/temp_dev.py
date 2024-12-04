#%%
from typing import List, Callable, Union, Any, TypeVar, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from models.base_vae import BaseVAE
from models.encoders import ConvEncoder
from models.decoders import ConvDecoder

#%%
if __name__ == "__main__":
    config = {
        'in_channels': 3,
        'latent_dim': 2,
        'hidden_dims': [32, 64, ],
        'width': 28, 'height': 28}


    dummy_input = torch.randn(7, *(config[k] for k in ['in_channels', 'width', 'height']))
    print("dummy_input.shape=", dummy_input.shape)
    input_channels = config['in_channels']
    hidden_dims = config['hidden_dims']
    latent_dim = config['latent_dim']
    encoder = ConvEncoder(input_channels=input_channels, hidden_dims=hidden_dims)

    encout = encoder(dummy_input)
    raw_encout_shape = encout.shape
    print("raw_encout_shape=", raw_encout_shape)

    encout = nn.Flatten(start_dim=1)(encout)
    
    flatten_dim = raw_encout_shape[1:].numel()
    print("flatten_dim=", flatten_dim)

    fc_mu = nn.Linear(flatten_dim, latent_dim)
    fc_var = nn.Linear(flatten_dim, latent_dim)

    z = fc_mu(encout)
    print("z.shape=", z.shape)

    decoder_input = nn.Linear(latent_dim, flatten_dim)(z)
    decoder_input = decoder_input.view(-1, *raw_encout_shape[1:])
    print("decoder_input.shape=", decoder_input.shape)

    decoder = ConvDecoder(hidden_dims=hidden_dims[::-1], output_channels=input_channels)

    raw_decoder_out = decoder(decoder_input)
    print("raw_decoder_out.shape=", raw_decoder_out.shape)