import torch.nn as nn

from .bottleneck import Bottleneck
from .conv2d import Conv2D
from .doubleConv2d import DoubleConv2D
from .downConv2d import DownConv2D
from .upConv2d import UpConv2D


class Unet2D(nn.Module):
    def __init__(
        self, in_channels: int, channels: int, frames: int, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        # input data
        self.input = DoubleConv2D(in_channels=in_channels, out_channels=channels)

        # encoder
        self.encoder_1 = DownConv2D(in_channels=channels, out_channels=2 * channels)
        self.encoder_2 = DownConv2D(in_channels=2 * channels, out_channels=4 * channels)
        self.encoder_3 = DownConv2D(in_channels=4 * channels, out_channels=8 * channels)

        # bottleneck
        self.bottleneck = Bottleneck(
            in_channels=8 * channels,
            out_channels=16 * channels,
            channels_bottleneck=frames,
        )

        # decoder
        self.decoder_3 = UpConv2D(in_channels=16 * channels, out_channels=8 * channels)
        self.decoder_2 = UpConv2D(in_channels=8 * channels, out_channels=4 * channels)
        self.decoder_1 = UpConv2D(in_channels=4 * channels, out_channels=2 * channels)
        self.decoder_input = UpConv2D(in_channels=2 * channels, out_channels=channels)

        # output
        self.last_conv2D = nn.Conv2d(
            in_channels=channels, out_channels=frames, kernel_size=1, stride=1
        )

    def forward(self, x) -> tuple:
        # input
        x_input = self.input(x)

        # encoder
        x_encoder_1 = self.encoder_1(x_input)
        x_encoder_2 = self.encoder_2(x_encoder_1)
        x_encoder_3 = self.encoder_3(x_encoder_2)

        # bottleneck
        x_bottleneck, x_up = self.bottleneck(x_encoder_3)

        # decoder
        x_decoder_3 = self.decoder_3(x_up, x_encoder_3)
        x_decoder_2 = self.decoder_2(x_decoder_3, x_encoder_2)
        x_decoder_1 = self.decoder_1(x_decoder_2, x_encoder_1)
        x_decoder_input = self.decoder_input(x_decoder_1, x_input)

        # output
        x_output = self.last_conv2D(x_decoder_input)

        return x_bottleneck, x_output
