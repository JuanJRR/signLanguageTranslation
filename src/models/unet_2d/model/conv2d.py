import torch.nn as nn


class Conv2D(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            padding_mode="zeros",
            groups=1,
            bias=True,
        )

        self.conv2d = nn.Sequential(
            conv, nn.BatchNorm2d(num_features=out_channels), nn.ReLU()
        )

    def forward(self, x):
        return self.conv2d(x)
