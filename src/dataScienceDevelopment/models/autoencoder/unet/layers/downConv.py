import torch.nn as nn

from .doubleConv import DoubleConv


class DownConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder_layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels=in_channels, out_channels=out_channels),
        )

    def forward(self, x):
        x_encoder_layer = self.encoder_layer(x)

        return x_encoder_layer
