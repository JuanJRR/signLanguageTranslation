import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from data.dataListGenc import DataListGeneratorClassifier
from etl.videoResampledDataIncrement import VideoResampledDataIncrement
from utils.filesCSV import FilesCSV
from utils.logger import Logger
from utils.videoHandling import VideoHandling

"""Constructor of the Dataset object, which Starts the directory that 
contains the video paths, the annotations file and transformations"""


class DatasetWordSLC(Dataset):
    """It allows you to create a data set, from a seed of items, or iterate through a file of items."""

    def __init__(
        self,
        annotations_dir: str,
        items_dir: str,
        video_units: int = 12,
        size_list: int = 255,
        video: dict = {"pixels": 90, "aspect_ratio": [16, 9], "color": "GRAY"},
        annotations_file: bool = False,
    ) -> None:
        # settings
        logger = Logger()
        self.log = logger.config_logging()

        self.video = VideoHandling(
            video["pixels"], video["aspect_ratio"], video["color"]
        )

        self.video_units = video_units
        self.video_resampled = VideoResampledDataIncrement(
            video_length=self.video_units
        )

        # transforms
        self.trans = transforms.Compose([transforms.ToTensor()])

        # upload information
        self.items_dir = items_dir
        self.size_list = size_list

        if annotations_file:
            self.id_file, self.label_file = FilesCSV.read_csv(annotations_dir)
        else:
            generator_data = DataListGeneratorClassifier(
                annotations_file=annotations_dir
            )

            list_generator = generator_data.generator(size_list=self.size_list)
            self.id_file = list_generator["ID"]
            self.label_file = list_generator["FIRMA"]

            del list_generator

            self.log.info("Built words dataset SLC")

    def __len__(self):
        return len(self.label_file)

    def __getitem__(self, idx) -> tuple:
        """It loads a video from your location and resamples it to a defined length,
            as well as scales it and sets its color scale. To return a list of frames and their label.
        :return: Tuple containing a list of frames and a text label
        :rtype: tuple
        """
        self.log.debug("Starting Loading, building and returning information ...")

        path_item = os.path.join(str(self.items_dir), str(self.id_file[idx]))
        frames = self.video.upload_video(filename=path_item)

        new_frames = self.video_resampled.videoResampled(frames=frames)
        new_frames = np.transpose(new_frames, (1, 2, 0))
        new_frames = self.trans(new_frames)

        del path_item, frames

        self.log.debug("Finished Loading, building and returning information")
        self.log.info("Data built and delivered")

        # return {"VIDEO": new_frames, "LABEL": self.label_file[idx]}
        return new_frames, str(self.label_file[idx])
