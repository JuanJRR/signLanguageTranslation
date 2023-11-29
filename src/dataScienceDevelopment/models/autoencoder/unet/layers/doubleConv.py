import torch.nn as nn

from .conv import Conv


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        conv_layer_1 = Conv(in_channels=in_channels, out_channels=out_channels)
        conv_layer_2 = Conv(in_channels=out_channels, out_channels=out_channels)

        self.double_conv = nn.Sequential(conv_layer_1, conv_layer_2)

    def forward(self, x):
        x_double_conv = self.double_conv(x)

        return x_double_conv
