import torch
import torch.nn as nn

from .doubleConv2d import DoubleConv2D


class UpConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.up_conv2D = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bicubic"),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=1,
                stride=1,
            ),
        )

        self.decoder = DoubleConv2D(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x1, x2):
        # print(x1.shape, x2.shape)
        x_upsample = self.up_conv2D(x1)
        # print(x_upsample.shape)

        # print(x2.shape)
        feature_maps = torch.cat([x2, x_upsample], dim=1)
        # print(feature_maps.shape)

        return self.decoder(feature_maps)
