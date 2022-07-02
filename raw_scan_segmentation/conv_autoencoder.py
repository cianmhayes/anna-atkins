from torch import nn

class ConvEncoder(nn.Module):
    def __init__(self, output_channels: int, kernel_size: int, input_channels: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                input_channels, output_channels, kernel_size, stride=2, padding=1
            ),
            nn.ReLU(True),
            nn.Conv2d(
                output_channels, output_channels * 2, kernel_size, stride=2, padding=1
            ),
            nn.BatchNorm2d(output_channels * 2),
            nn.ReLU(True),
            nn.Conv2d(
                output_channels * 2,
                output_channels * 4,
                kernel_size,
                stride=2,
                padding=0,
            ),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.encoder(x)


class ConvDecoder(nn.Module):
    def __init__(self, output_channels: int, kernel_size: int, input_channels: int = 3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                output_channels * 4,
                output_channels * 2,
                kernel_size,
                stride=2,
                padding=0,
                output_padding=0,
            ),
            nn.BatchNorm2d(output_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                output_channels * 2,
                output_channels,
                kernel_size,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                output_channels,
                input_channels,
                kernel_size,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.decoder(x)


class ConvAutoEncoder(nn.Module):
    def __init__(self, output_channels: int, kernel_size: int, input_channels: int = 3):
        super().__init__()
        self.encoder = ConvEncoder(output_channels, kernel_size, input_channels)
        self.decoder = ConvDecoder(output_channels, kernel_size, input_channels)

    def forward(self, x):
        embedding = self.encoder(x)
        reconstructed = self.decoder(embedding)
        return embedding, reconstructed
