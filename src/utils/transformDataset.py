import torch
from torchvision.transforms import v2


class TransformDataset:
    @staticmethod
    def original_image():
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        return transform

    @staticmethod
    def image_rescaling(pixel_density: int = 128):
        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=(pixel_density, pixel_density * 2), antialias=True),
            ]
        )

        return transform
