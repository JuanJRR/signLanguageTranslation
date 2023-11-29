import os

import pandas as pd
from torch.utils.data import Dataset
from torchvision import io


class GestureSegmentationDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        annotation_path,
        maximum_items: int = 0,
        transform=None,
    ) -> None:
        super().__init__()

        # setting
        self.data_path = data_path
        self.transform = transform

        # load data list
        metadata_file = pd.read_csv(annotation_path, sep=",")
        self.path_images = metadata_file["images"]
        self.path_masks = metadata_file["masks"]

        if maximum_items > 0 and maximum_items < len(self.path_images):
            self.path_images = self.path_images[0:maximum_items]
            self.path_masks = self.path_masks[0:maximum_items]

        del metadata_file

    def __len__(self):
        if len(self.path_images) == len(self.path_masks):
            return len(self.path_images)

    def __getitem__(self, idx):
        path_image = os.path.join(self.data_path, str(self.path_images[idx]))
        path_mask = os.path.join(self.data_path, str(self.path_masks[idx]))

        image = io.read_image(path=path_image, mode=io.ImageReadMode.RGB)
        mask = io.read_image(path=path_mask, mode=io.ImageReadMode.GRAY)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
