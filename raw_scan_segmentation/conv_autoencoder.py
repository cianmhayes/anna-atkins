import torch
from torch import nn
from typing import NamedTuple

class PatchConvAutoEncoderParameters(NamedTuple):
    kernel_size: int
    hidden_layers: int
    initial_channels: int
    final_channels: int
    input_channels: int = 3
    

class ConvEncoder(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        hidden_layers: int,
        initial_channels: int,
        final_channels: int,
        input_channels: int = 3,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                input_channels, initial_channels, kernel_size, stride=int(kernel_size / 2)
            ),
            nn.ReLU(True),
        )
        last_channel_size = initial_channels
        for i in range(hidden_layers + 1):
            self.encoder.append(nn.Conv2d(last_channel_size, final_channels, 1))
            self.encoder.append(nn.ReLU(True))
            self.encoder.append(nn.BatchNorm2d(final_channels))
            last_channel_size = final_channels

    def forward(self, x):
        return self.encoder(x)


class ConvDecoder(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        hidden_layers: int,
        initial_channels: int,
        final_channels: int,
        input_channels: int = 3,
    ):
        super().__init__()
        self.decoder = nn.Sequential()
        for i in range(hidden_layers + 1, 0, -1):
            output_channels = final_channels
            if i == 1:
                output_channels = initial_channels
            self.decoder.append(nn.Conv2d(final_channels, output_channels, 1))
            self.decoder.append(nn.ReLU(True))
            self.decoder.append(nn.BatchNorm2d(output_channels))
        self.decoder.append(
            nn.ConvTranspose2d(
                initial_channels, input_channels, kernel_size, stride=int(kernel_size / 2)
            )
        )
        self.decoder.append(nn.ReLU(True))

    def forward(self, x):
        return self.decoder(x)


class ConvAutoEncoder(nn.Module):
    def __init__(
        self,
        parameters:PatchConvAutoEncoderParameters
    ):
        super().__init__()
        self.encoder = ConvEncoder(
            parameters.kernel_size, parameters.hidden_layers, parameters.initial_channels, parameters.final_channels, parameters.input_channels
        )
        self.decoder = ConvDecoder(
            parameters.kernel_size, parameters.hidden_layers, parameters.initial_channels, parameters.final_channels, parameters.input_channels
        )

    def forward(self, x):
        embedding = self.encoder(x)
        reconstructed = self.decoder(embedding)
        return embedding, reconstructed

def load_pretrained_autoencoder_from_checkpoint(parameters:PatchConvAutoEncoderParameters, path:str) -> ConvAutoEncoder:
    model = ConvAutoEncoder(parameters)
    model.load_state_dict(torch.load(path))
    return model