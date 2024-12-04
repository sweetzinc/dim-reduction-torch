# %%
import torch
import torch.nn as nn
from models.encoders import LinearEncoder, ConvEncoder


class ConvDecoder(nn.Module):
    def __init__(self, hidden_dims=[128,64,32], output_channels=3):
        super().__init__()
        self.hidden_dims = hidden_dims

        modules = []
        # Build Decoder Layers
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               output_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.Sigmoid()  
        )

    def forward(self, z):
        x_hat = self.decoder(z)
        x_hat = self.final_layer(x_hat)
        return x_hat

# %%
if __name__ == "__main__":

    vae_config = {
        'in_channels': 64,
        'latent_dim': 2,
        'hidden_dims': [32, 64, ],
        'width': 8, 'height': 8}
    hidden_dims = vae_config['hidden_dims'][::-1]

    dummy_input = torch.randn(
        7, *(vae_config[k] for k in ['in_channels', 'width', 'height']))
    print("dummy_input.shape=", dummy_input.shape)
 
    decoder = ConvDecoder(hidden_dims=hidden_dims, output_channels=3)
    decoder_output = decoder(dummy_input)

    print("decoder_output.shape=", decoder_output.shape)

#%%
if __name__ == "__main__":
    # input_width = 28
    # padding = 0
    # kernel_size = 3
    # stride = 1
    # dilation = 1
    # output_width = (input_width + 2*padding - dilation*(kernel_size-1) - 1)
    # output_width = output_width//stride + 1
    # print("output_width=", output_width)

    # conv = nn.Conv2d(1, 32, kernel_size=kernel_size, stride=stride, padding=padding)
    # x = torch.randn(7, 1, input_width, input_width)
    # y = conv(x)
    # print("y.shape=", y.shape)

    # For fixing the output width to be the same as the input width
    # fix stride and dilation to 1, find kernel size
    padding = 0
    desired_output_width = 28 
    kernel_size = decoder_output.shape[-1] - desired_output_width + 1 + 2*padding
    print("kernel_size=", kernel_size)

    conv = nn.Conv2d(1, 32, kernel_size=kernel_size, stride=1, padding=padding)
    x = torch.randn(7, 1, decoder_output.shape[-1], decoder_output.shape[-1])
    y = conv(x)
    print("y.shape=", y.shape)
#%%
class LinearDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim=None):
        """latent_dim: Size of the latent space, input to the decoder."""
        super().__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(latent_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU())
            latent_dim = h_dim
        if output_dim is not None:
            # Final layer maps back to original input dimension
            layers.append(nn.Linear(latent_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class LinearAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, **kwargs):
        super().__init__()
        self.encoder = LinearEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder = LinearDecoder(latent_dim, hidden_dims[::-1], input_dim)

    def forward(self, x, return_latent=False):
        latent = self.encoder(x)  # Latent representation
        if return_latent:
            return latent
        reconstructed = self.decoder(latent)
        return reconstructed


class MultiChannelLinearAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, channels=3):
        """
        Autoencoder handling multiple channels with separate encoders and decoders.

        Args:
            input_dim (int): Number of input features per channel.
            hidden_dims (list of int): Number of units in each encoder layer.
            latent_dim (int): Size of the latent space.
            channels (int): Number of channels (e.g., 3 for RGB).
        """
        super().__init__()
        self.channels = channels

        # Create separate encoders and decoders for each channel
        self.encoders = nn.ModuleList([
            LinearEncoder(input_dim, hidden_dims, latent_dim) for _ in range(channels)
        ])
        self.decoders = nn.ModuleList([
            LinearDecoder(latent_dim, hidden_dims[::-1], input_dim) for _ in range(channels)
        ])

    def forward(self, x, return_latent=False):
        """
        Forward pass of the multi-channel autoencoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, features, channels).
            return_latent (bool): If True, return latent representations instead of reconstructed output.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Reconstructed input or latent representations.
        """
        batch_size, features, channels = x.size()
        assert channels == self.channels, f"Expected {self.channels} channels, got {channels}"

        # Initialize lists to store latent representations and reconstructions
        latent_reps = []
        reconstructions = []

        # Iterate over each channel
        for c in range(channels):
            # Extract data for channel c: shape (batch_size, features)
            channel_data = x[:, :, c]

            # Encode
            latent = self.encoders[c](channel_data)
            latent_reps.append(latent)

            if not return_latent:
                # Decode
                reconstructed = self.decoders[c](latent)
                reconstructions.append(reconstructed)

        if return_latent:
            # Stack latent representations: list of (batch_size, latent_dim) -> (batch_size, latent_dim, channels)
            latent_stack = torch.stack(latent_reps, dim=2)
            return latent_stack  # Shape: (batch_size, latent_dim, channels)
        else:
            # Stack reconstructions: list of (batch_size, features) -> (batch_size, features, channels)
            reconstructed_stack = torch.stack(reconstructions, dim=2)
            # Shape: (batch_size, features, channels)
            return reconstructed_stack
# %%
# if __name__ == "__main__":
#     # Configuration
#     input_dim = 100
#     encoder_dims = [64, 32, 16]
#     latent_dim = 8

#     # Model
#     model = LinearAutoencoder(input_dim, encoder_dims, latent_dim)

#     # Dummy data
#     x = torch.randn(10, input_dim)  # Batch size of 10


#     # Get the latent representation
#     latent = model(x, return_latent=True)
#     print(f"Latent shape: {latent.shape}")
#     # Get the reconstructed input
#     reconstructed = model(x, return_latent=False)
#     print(f"Reconstructed shape: {reconstructed.shape}")
# %%
if False:  # __name__ == "__main__":
    # Configuration
    input_dim = 100          # Number of input features per channel
    encoder_dims = [64, 32, 16]  # Encoder layer sizes
    latent_dim = 8           # Latent space size
    channels = 3             # Number of channels (e.g., RGB)

    # Instantiate the model
    model = MultiChannelLinearAutoencoder(
        input_dim=input_dim, hidden_dims=encoder_dims, latent_dim=latent_dim, channels=channels)
    print(model)

    # Test with different batch sizes
    for batch_size in [10, 32]:
        # Dummy data: shape (batch_size, features, channels)
        x = torch.randn(batch_size, input_dim, channels)

        # Get latent representations
        latent = model(x, return_latent=True)
        # Expected: (batch_size, latent_dim, channels)
        print(f"Batch size: {batch_size}, Latent shape: {latent.shape}")

        # Get reconstructed input
        reconstructed = model(x, return_latent=False)
        # Expected: (batch_size, features, channels)
        print(
            f"Batch size: {batch_size}, Reconstructed shape: {reconstructed.shape}")
# %%
