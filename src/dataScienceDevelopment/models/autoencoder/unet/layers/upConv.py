import torch
import torch.nn as nn

from .doubleConv import DoubleConv


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bicubic"),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=1,
                stride=1,
            ),
        )

        self.decoder_layer = DoubleConv(
            in_channels=in_channels, out_channels=out_channels
        )

    def forward(self, x1, x2):
        x_upsample = self.upsample(x1)
        x_feature_maps = torch.cat([x2, x_upsample], dim=1)
        x_decoder_layer = self.decoder_layer(x_feature_maps)

        return x_decoder_layer
