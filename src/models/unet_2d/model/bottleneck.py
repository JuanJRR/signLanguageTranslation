import torch.nn as nn

from .conv2d import Conv2D


class Bottleneck(nn.Module):
    def __init__(
        self, in_channels, out_channels, channels_bottleneck, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2D(in_channels=in_channels, out_channels=channels_bottleneck),
        )

        self.conv_2d = Conv2D(
            in_channels=channels_bottleneck, out_channels=out_channels
        )

    def forward(self, x):
        x_bottleneck = self.bottleneck(x)
        x_upConv = self.conv_2d(x_bottleneck)

        return x_bottleneck, x_upConv
