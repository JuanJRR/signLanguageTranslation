import torch.nn as nn

from .downConv import DownConv


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.bottleneck_layer = DownConv(
            in_channels=in_channels, out_channels=out_channels
        )

    def forward(self, x):
        x_bottleneck_layer = self.bottleneck_layer(x)

        return x_bottleneck_layer
