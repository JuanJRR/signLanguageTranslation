import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from dataScienceDevelopment.models.autoencoder.unet.model_unet import Unet
from dataSciencePipeline.pipTrainingUnet import PipTrainingUnet
from dataWrangling.gestureSegmentationDataset import GestureSegmentationDataset
from utils.logger import Logger
from utils.transformDataset import TransformDataset

if __name__ == "__main__":
    # Settings
    load_dotenv()

    # Logging
    logger = Logger()
    log = logger.config_logging()

    # Execution App
    log.critical("Star App ...")
    log.warning("Loading settings ...")

    # Environment Variables
    log.info("Settings:Environment Variables")
    seed_use = os.getenv("seed_use")

    # Seed
    log.info("Settings:Seed")

    if seed_use == "TRUE":
        seed = 1117
        np.random.seed(seed)
        torch.manual_seed(seed)
        log.info("Defined seed: {}".format(seed))
    else:
        log.info("Random seed")

    # Execution App
    log.warning("Starting main program loop ...")

    # Training pipeline unet segmentation
    pip_training_unet = PipTrainingUnet(
        path_save_experiment="models/unet_segmentation",
        pixel_density=128,
        model_dimensions={"in_channels": 3, "out_channels": 2, "channels": 4},
    )

    pip_training_unet.training(
        name="ex1_unetSegmentationPTrain",
        data_path="data/raw/human_segmentation_dataset",
        path_data_training="data/raw/human_segmentation_dataset/df_training.csv",
        path_data_evaluation="data/raw/human_segmentation_dataset/df_test.csv",
        batch_size=32,
        flag_scheduler=True,
        epochs=5,
    )

    # -------------

    # ges = GestureSegmentationDataset(
    #     data_path="data/raw/human_segmentation_dataset",
    #     annotation_path="data/raw/human_segmentation_dataset/df.csv",
    #     maximum_items=100,
    #     transform=TransformDataset.image_rescaling(pixel_density=128),
    # )

    # train_loader = DataLoader(ges, batch_size=32, shuffle=True)
    # imgs, masks = next(iter(train_loader))
    # print(imgs.shape, masks.shape)

    # def plot_mini_batch(imgs, masks):
    #     plt.figure(figsize=(20, 10))
    #     for i in range(32):
    #         plt.subplot(4, 8, i + 1)
    #         img = imgs[i, ...].permute(1, 2, 0).numpy()
    #         mask = masks[i, ...].permute(1, 2, 0).numpy()
    #         plt.imshow(img)
    #         plt.imshow(mask, alpha=0.5)

    #         plt.axis("Off")
    #     plt.tight_layout()
    #     plt.show()

    # plot_mini_batch(imgs, masks)
    # ------------------------

    # def plot_mini_batch_bt(imgs):
    #     img = imgs[0]
    #     print(img.shape)

    #     plt.figure(figsize=(20, 16))
    #     for i in range(img.shape[0]):
    #         plt.subplot(8, 8, i + 1)
    #         v_img = img[i].detach().numpy()
    #         # mask = masks[i, ...].permute(1, 2, 0).detach().numpy()
    #         plt.imshow(v_img)
    #         # plt.imshow(mask, alpha=0.5)

    #         plt.axis("Off")
    #     plt.tight_layout()
    #     plt.show()

    # def plot_mini_batch_mask(masks):
    #     masks = masks[0].detach().numpy()
    #     plt.imshow(masks[0])
    #     plt.imshow(masks[1])
    #     plt.show()

    # # point = torch.load("models/unet/checkpoints_e2_pretrainingPeopleE1_acc55.pt")
    # # c_point = point["model_state_dict"]
    # model = Unet(in_channels=3, out_channels=2, channels=4)
    # model.load_state_dict(
    #     torch.load("models/unet_segmentation/ex1_unetSegmentationPTrain_model.pt")
    # )
    # # model.load_state_dict(c_point)
    # model.eval()
    # train_features, train_labels = ges[17]
    # # print(train_features.unsqueeze(0).shape)
    # bottleneck, output = model(train_features.unsqueeze(0))
    # plot_mini_batch_bt(bottleneck)
    # plot_mini_batch_mask(output)

    # # Fin App
    # log.warning("End main program loop ...")
    # log.critical("Stop App ...")
