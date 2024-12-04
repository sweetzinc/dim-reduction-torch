#%%
from typing import List, Callable, Union, Any, TypeVar, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from models.base_vae import BaseVAE
from models.encoders import ConvEncoder
from models.decoders import ConvDecoder
#%%
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
            self.raw_encout_shape= self.encoder(torch.zeros(1, in_channels, width, height)).shape

        self.flatten_dim = self.raw_encout_shape[1:].numel()
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_var = nn.Linear(self.flatten_dim, latent_dim)


        self.decoder_fc = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = ConvDecoder(hidden_dims=hidden_dims[::-1], output_channels=in_channels)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_hat = self.decoder_fc(z)
        x_hat = x_hat.view(-1, *self.raw_encout_shape[1:])
        x_hat = self.decoder(x_hat)
        return x_hat

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, x, mu, log_var

    def loss_function(self, model_output, **kwargs) -> dict:
        x_hat, x, mu, log_var = model_output

        # Reconstruction loss
        recons_loss = F.mse_loss(x_hat, x, reduction='sum')

        # KL Divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        if kwargs.get('kld_weight', 1) != 1:
            kl_div = kl_div * kwargs['kld_weight']

        loss = recons_loss + kl_div
        return {'loss': loss, 
                'Reconstruction_Loss':recons_loss.detach(), 
                'KLD':-kl_div.detach()}
    
# %%
if __name__ == "__main__":
    config = {
        'in_channels': 3,
        'latent_dim': 2,
        'hidden_dims': [32, 64, ],
        'width': 32, 'height': 32}
    
    vae = VanillaVAE(**config)
    # print(vae)

    dummy_input = torch.randn(7, *(config[k] for k in ['in_channels', 'width', 'height']))
    print("dummy_input.shape=", dummy_input.shape)

    x_hat, x, mu, log_var = vae(dummy_input)
    print("x_hat.shape=", x_hat.shape)
    print("x.shape=", x.shape)
    print("mu.shape=", mu.shape)
    print("log_var.shape=", log_var.shape)

    loss = vae.loss_function([x_hat, x, mu, log_var])
    print("loss: ", loss)
# %%
