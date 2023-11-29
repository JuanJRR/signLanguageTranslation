import torch.nn as nn

from .layers.bottleneck import Bottleneck
from .layers.doubleConv import DoubleConv
from .layers.downConv import DownConv
from .layers.upConv import UpConv


class Unet(nn.Module):
    def __init__(
        self, in_channels: int, channels: int, out_channels: int, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        # Encoder
        self.encoder_1 = DoubleConv(in_channels=in_channels, out_channels=channels)
        self.encoder_2 = DownConv(in_channels=channels, out_channels=2 * channels)
        self.encoder_3 = DownConv(in_channels=2 * channels, out_channels=4 * channels)
        self.encoder_4 = DownConv(in_channels=4 * channels, out_channels=8 * channels)

        # Bottleneck
        self.bottleneck = Bottleneck(
            in_channels=8 * channels, out_channels=16 * channels
        )

        # Decoder
        self.decoder_4 = UpConv(in_channels=16 * channels, out_channels=8 * channels)
        self.decoder_3 = UpConv(in_channels=8 * channels, out_channels=4 * channels)
        self.decoder_2 = UpConv(in_channels=4 * channels, out_channels=2 * channels)
        self.decoder_1 = UpConv(in_channels=2 * channels, out_channels=channels)

        # Output
        self.output = nn.Conv2d(
            in_channels=channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x) -> tuple:
        # Encoder
        x_encoder_1 = self.encoder_1(x)
        x_encoder_2 = self.encoder_2(x_encoder_1)
        x_encoder_3 = self.encoder_3(x_encoder_2)
        x_encoder_4 = self.encoder_4(x_encoder_3)

        # Bottleneck
        x_bottleneck = self.bottleneck(x_encoder_4)

        # Decoder
        x_decoder_4 = self.decoder_4(x_bottleneck, x_encoder_4)
        x_decoder_3 = self.decoder_3(x_decoder_4, x_encoder_3)
        x_decoder_2 = self.decoder_2(x_decoder_3, x_encoder_2)
        x_decoder_1 = self.decoder_1(x_decoder_2, x_encoder_1)

        # Output
        x_output = self.output(x_decoder_1)

        return x_bottleneck, x_output
