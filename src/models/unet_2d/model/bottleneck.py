import torch.nn as nn

from .conv2d import Conv2D


class Bottleneck(nn.Module):
    def __init__(
        self, in_channels, out_channels, channels_bottleneck, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.conv_bottleneck_down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2D(in_channels=in_channels, out_channels=out_channels),
        )

        self.bottleneck = Conv2D(
            in_channels=out_channels, out_channels=channels_bottleneck
        )

        self.conv_bottleneck_up = Conv2D(
            in_channels=channels_bottleneck, out_channels=out_channels
        )

    def forward(self, x):
        x_down = self.conv_bottleneck_down(x)
        x_bottleneck = self.bottleneck(x_down)
        x_up = self.conv_bottleneck_up(x_bottleneck)

        return x_bottleneck, x_up
