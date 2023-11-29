import os

import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class PretrainingPeopleDataset(Dataset):
    def __init__(
        self, path_data: str, transforms: bool = True, pixels: int = 128
    ) -> None:
        super().__init__()

        # Setting
        self.path_data = path_data
        self.transforms = transforms
        self.pixels = pixels

        # Transformations
        self.trans = T.ToTensor()
        self.img_transforms = T.Compose(
            [T.Resize([self.pixels, self.pixels * 2]), T.ToTensor()]
        )

        # Load image metadata
        self.images_path = []
        # self.labels = []

        folders = os.listdir(self.path_data)

        for index_f, folder in enumerate(folders):
            subdirectories = os.path.join(self.path_data, folder)
            files = os.listdir(subdirectories)
            for index_img, image in enumerate(files):
                img = os.path.join(subdirectories, image)
                self.images_path.append(img)
                # self.labels.append(folder)

    def __len__(self):
        if len(self.images_path) is not None:
            return len(self.images_path)

    def __getitem__(self, idx):
        # label = self.labels[idx]
        img = Image.open(self.images_path[idx])
        img = img.convert("L")

        if self.transforms:
            img = self.img_transforms(img)
        else:
            img = self.trans(img)

        return img, ""
