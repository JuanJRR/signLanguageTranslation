import torch.nn as nn

from .conv2d import Conv2D


class DoubleConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        conv_1 = Conv2D(in_channels=in_channels, out_channels=out_channels)
        conv_2 = Conv2D(in_channels=out_channels, out_channels=out_channels)

        self.double_conv = nn.Sequential(conv_1, conv_2)

    def forward(self, x):
        return self.double_conv(x)
