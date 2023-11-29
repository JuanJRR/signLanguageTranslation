import torch.nn as nn
from torch.utils.data import DataLoader

from dataScienceDevelopment.models.autoencoder.unet.model_unet import Unet
from dataScienceDevelopment.optimisers.optimUnet import OptimUnet
from dataScienceDevelopment.schedulers.schedulerUnet import SchedulerUnet
from dataScienceDevelopment.trainingLoops.unetTraining import UnetTraining
from dataWrangling.gestureSegmentationDataset import GestureSegmentationDataset
from utils.transformDataset import TransformDataset


class PipTrainingUnet:
    def __init__(
        self,
        path_save_experiment: str,
        pixel_density: int = 128,
        model_dimensions: dict = {"in_channels": 1, "out_channels": 1, "channels": 16},
    ) -> None:
        # settings
        self.path_save_experiment = path_save_experiment

        # settings data
        self.pixel_density = pixel_density

        # Model
        self.model_dimensions = model_dimensions
        self.model = Unet(
            in_channels=self.model_dimensions["in_channels"],
            out_channels=self.model_dimensions["out_channels"],
            channels=self.model_dimensions["channels"],
        )

        # optim
        self.optim = OptimUnet(model=self.model)

    def __load_data(
        self,
        data_path: str,
        path_data_training: str,
        path_data_evaluation=None,
        batch_size: int = 32,
    ):
        # data training
        data_training = GestureSegmentationDataset(
            annotation_path=path_data_training,
            data_path=data_path,
            maximum_items=0,
            transform=TransformDataset.image_rescaling(
                pixel_density=self.pixel_density
            ),
        )

        dt_training = DataLoader(
            dataset=data_training, batch_size=batch_size, shuffle=True
        )

        # data evaluation
        dt_evaluation = None
        if path_data_evaluation != "":
            data_evaluation = GestureSegmentationDataset(
                annotation_path=path_data_evaluation,
                data_path=data_path,
                maximum_items=0,
                transform=TransformDataset.image_rescaling(
                    pixel_density=self.pixel_density
                ),
            )

            dt_evaluation = DataLoader(
                dataset=data_evaluation, batch_size=batch_size, shuffle=True
            )
        else:
            dt_evaluation = None

        return dt_training, dt_evaluation

    def training(
        self,
        name,
        data_path: str,
        path_data_training: str,
        path_data_evaluation=None,
        lr: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        flag_scheduler: bool = False,
        max_lr: float = 0.1,
    ):
        # Data
        dt_training, dt_evaluation = self.__load_data(
            data_path,
            path_data_training=path_data_training,
            path_data_evaluation=path_data_evaluation,
            batch_size=batch_size,
        )

        # optimizer
        loss_function = nn.CrossEntropyLoss()
        optimizer = self.optim.adam(lr=lr)

        # scheduler
        if flag_scheduler:
            self.schedulers = SchedulerUnet(optimizer=optimizer, epochs=epochs)
            scheduler = self.schedulers.scheduler_OneCycleLR(
                steps_per_epoch=len(dt_training), max_lr=max_lr
            )
        else:
            scheduler = None

        # loop training
        training_model = UnetTraining(
            model=self.model,
            optimizer=optimizer,
            loss_function=loss_function,
            scheduler=scheduler,
            data_training=dt_training,
            data_validation=dt_evaluation,
            epochs=epochs,
            path_save_experiment=self.path_save_experiment,
            early_stop={"stop": True, "patience": 3},
            details={"report": 1, "iterations": 5},
            model_dimensions=self.model_dimensions,
        )

        training_model.training_loop(name=name)
