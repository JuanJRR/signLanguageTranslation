import torch.nn as nn

from .doubleConv2d import DoubleConv2D


class DownConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv2D(in_channels=in_channels, out_channels=out_channels),
        )

    def forward(self, x):
        return self.encoder(x)
