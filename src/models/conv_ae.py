#%%
import torch
import torch.nn as nn
from models.encoders import ConvEncoder
from models.decoders import ConvDecoder


class Autoencoder(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_dims):
        """
        Combines the Encoder and Decoder into an Autoencoder.
        Args:
            input_channels (int): Number of channels in the input image.
            output_channels (int): Number of channels in the output image.
            hidden_dims (list): List of integers for the hidden dimensions of the encoder.
        """
        super(Autoencoder, self).__init__()
        self.encoder = ConvEncoder(input_channels, hidden_dims)
        self.decoder = ConvDecoder(hidden_dims[::-1], output_channels,)

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed
    
if __name__ == "__main__":
    # Configuration
    input_channels = 3  # For RGB images
    output_channels = 3  # Output should match input channels
    hidden_dims = [32, 64, 128, 256]  # Example hidden dims

    # Model
    model = Autoencoder(input_channels, output_channels, hidden_dims)
    print(model)

    # Test with dummy data
    x = torch.randn(8, input_channels, 128, 128)  # Batch of 8, 128x128 RGB images
    z = model.encoder(x)
    print("z.shape:", z.shape)  # Should match the last hidden dim
    reconstructed = model(x)
    print("reconstructed.shape: ", reconstructed.shape)  # Should match input shape