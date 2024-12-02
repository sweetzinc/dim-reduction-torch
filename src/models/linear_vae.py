#%%
from typing import List, Tuple

# PyTorch
import torch
from torch import nn
import torch.nn.functional as F
from models.encoders import LinearEncoder
from models.decoders import LinearDecoder

class LinearVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = None,
        **kwargs
    ) -> None: 
        super().__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        # self.in_channels = in_channels
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64, 32]
        self.hidden_dims = hidden_dims

        # Encoder
        self.encoder = LinearEncoder(input_dim, hidden_dims)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        self.decoder = LinearDecoder(latent_dim, hidden_dims[::-1])
        self.final_layer = nn.Sequential(*[nn.Linear(hidden_dims[0], input_dim),
                                          nn.Sigmoid()])

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        reconstruction = self.decoder(z)
        reconstruction = self.final_layer(reconstruction)
        return reconstruction

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), x, mu, log_var

    def loss_function(self, recons, x, mu, log_var, kld_weight=0.005, **kwargs) -> dict:
        recons_loss = F.mse_loss(recons, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self, num_samples: int, device: str, **kwargs) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(x)[0]

#%%
if __name__ == "__main__":
    # Example setup for LinearVAE
    vae_config = {
        'input_dim': 784,  # 28x28 flattened
        'latent_dim': 2,
        'hidden_dims': [512, 128, 32, 8]
    }

    vae_model = LinearVAE(**vae_config)

    # Create a dummy input (batch size 4, 784 flattened input)
    dummy_input = torch.randn(4, vae_config['input_dim'])

    # Perform a forward pass
    output = vae_model(dummy_input)
    print("output[0].shape =", output[0].shape)  # Reconstructed output
    print("output[1].shape =", output[1].shape)  # Original input
    print("output[2].shape =", output[2].shape)  # Mu
    print("output[3].shape =", output[3].shape)  # Log Var
# %%
