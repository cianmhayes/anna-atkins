import torch
from torch import Tensor, nn
from conv_autoencoder import ConvEncoder
from dataclasses import dataclass
from typing import NamedTuple

class SegmenterParameters(NamedTuple):
    original_patch_size: int
    patch_encoding_size: int
    edge_detection_channels: int
    output_classes: int

class Segmenter(nn.Module):
    def __init__(
        self,
        patch_encoder: ConvEncoder,
        parameters: SegmenterParameters
    ):
        super().__init__()
        self.patch_encoder = patch_encoder
        self.original_patch_size = parameters.original_patch_size
        self.patch_encoding_size = parameters.patch_encoding_size
        self.edge_detection_channels = parameters.edge_detection_channels
        self.output_classes = parameters.output_classes
        self.kernel = nn.Sequential(
            nn.Conv2d(
                self.patch_encoding_size, self.edge_detection_channels, 3, stride=1, padding=1
            ),
            nn.ReLU(True),
            nn.BatchNorm2d(self.edge_detection_channels),
        )
        self.patch_unpacker = nn.Sequential(
            nn.ConvTranspose2d(
                self.edge_detection_channels,
                self.output_classes,
                self.original_patch_size,
                stride=int(self.original_patch_size / 2),
            ),
            nn.Softmax(1), # batch, channels, height, width
        )

    def forward(self, x):
        encoded_patches = self.patch_encoder(x)
        aggregated = self.kernel(encoded_patches)
        unpacked = self.patch_unpacker(aggregated)
        return unpacked
