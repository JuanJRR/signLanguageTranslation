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
        video_units: int = 30,
        size_list: int = 255,
        video: dict = {"pixels": 90, "aspect_ratio": [16, 9], "color": "GRAY"},
        use_annotation_list: bool = False,
        ram_preload: bool = True,
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

        # generador
        self.use_annotation_list = use_annotation_list
        self.id_file = []
        self.label_file = []
        self.keys = []

        if self.use_annotation_list:
            self.id_file, self.label_file = FilesCSV.read_csv(annotations_dir)
        else:
            generator_data = DataListGeneratorClassifier(
                annotations_file=annotations_dir
            )

            list_generator, keys = generator_data.generator(size_list=self.size_list)
            self.id_file = list_generator["ID"]
            self.label_file = list_generator["FIRMA"]
            self.keys = keys

            del list_generator, keys, generator_data

        # precarga
        self.ram_preload = ram_preload
        self.preload_frames = []

        if self.ram_preload and self.use_annotation_list:
            for index, item in enumerate(self.id_file):
                path_item = os.path.join(str(self.items_dir), str(item))
                frame_video = self.video.upload_video(filename=path_item)
                self.preload_frames.append(frame_video)
        elif self.ram_preload and self.use_annotation_list == False:
            id_file, label_file = FilesCSV.read_csv(annotations_dir)

            for index, item in enumerate(id_file):
                path_item = os.path.join(str(self.items_dir), str(item))
                frame_video = self.video.upload_video(filename=path_item)
                self.preload_frames.append(frame_video)

            del id_file, label_file

        self.log.info("Built words dataset SLC")

    def __len__(self):
        if self.ram_preload and self.use_annotation_list:
            assert len(self.label_file) == len(self.id_file) == len(self.preload_frames)
            return len(self.label_file)
        elif self.ram_preload and self.use_annotation_list == False:
            assert len(self.label_file) == len(self.id_file) == len(self.keys)
            return len(self.label_file)
        elif self.ram_preload == False:
            assert len(self.label_file) == len(self.id_file)
            return len(self.label_file)

    def __getitem__(self, idx) -> tuple:
        """It loads a video from your location and resamples it to a defined length,
            as well as scales it and sets its color scale. To return a list of frames and their label.
        :return: Tuple containing a list of frames and a text label
        :rtype: tuple
        """
        self.log.debug("Starting Loading, building and returning information ...")

        if self.ram_preload:
            if self.use_annotation_list:
                new_frames = self.video_resampled.videoResampled(
                    frames=self.preload_frames[idx]
                )
                new_frames = np.transpose(new_frames, (1, 2, 0))
                new_frames = self.trans(new_frames)
            else:
                index_frame = self.keys[idx]
                bbb = self.preload_frames
                new_frames = self.video_resampled.videoResampled(
                    frames=self.preload_frames[index_frame]
                )
                new_frames = np.transpose(new_frames, (1, 2, 0))
                new_frames = self.trans(new_frames)
        else:
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
        # return "ho", "la"
