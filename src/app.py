import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from torch import nn
from torch.utils.data import DataLoader

from dataWrangling.datasetWordSLC import DatasetWordSLC
from models.unet_2d.unet2d import Unet2D
from models.unet_2d.upConv2d import UpConv2D
from utils.logger import Logger

if __name__ == "__main__":
    # Settings
    load_dotenv()

    # Logging
    logger = Logger()
    log = logger.config_logging()

    # Execution App
    log.info("Star App ...")
    log.info("Loading settings ...")

    # Environment Variables
    log.info("Settings:Environment Variables")
    seed_use = os.getenv("seed_use")

    # Seed
    log.info("Settings:Seed")

    if seed_use == "TRUE":
        seed = 1
        np.random.seed(seed)
        torch.manual_seed(seed)
        log.info("Defined seed: {}".format(seed))
    else:
        log.info("Random seed")

    # Execution App
    log.info("Starting main program loop ...")

    # a = torch.randn(1, 30, 128, 256)
    # b = torch.randn(1, 30, 8, 14)
    # c = torch.rand(1, 30, 16, 28)

    # up = UpConv2D(in_channels=30, out_channels=30)
    # print(up(b, c).shape)
    # print(.shape)

    # conv = nn.Conv2d(
    #     in_channels=30, out_channels=30, kernel_size=3, padding=1, bias=False
    # )
    # print(a.shape)
    # b = conv(a)
    # print(b.shape)

    # m = Unet2D(in_channels=30, channels=30, frames=30)
    # # print(m)
    # bottleneck, output = m(a)
    # print(bottleneck.shape, output.shape)

    # rnn = nn.LSTM(10, 20, 2)
    # print(rnn)
    # input_1 = torch.randn(5, 3, 100)
    # input_2 = torch.randn(50, 30, 80)
    # # print(input.shape)
    # # h0 = torch.randn(2, 3, 20)
    # # c0 = torch.randn(2, 3, 20)
    # output, (hn, cn) = rnn(input_1)
    # output, (hn, cn) = rnn(input_2)
    # print(output.shape)

    # TD = DatasetWordSLC(
    #     annotations_file=False,
    #     annotations_dir="data/raw/10_words_3_people/000_10_words_3_people.csv",
    #     items_dir="data/raw/10_words_3_people/",
    #     video_units=30,
    #     size_list=10,
    #     video={"pixels": 128, "aspect_ratio": [16, 9], "color": "GRAY"},
    # )

    # TD_1 = DatasetWordSLC(
    #     annotations_file=False,
    #     annotations_dir="data/raw/10_words_3_people/000_10_words_3_people.csv",
    #     items_dir="data/raw/10_words_3_people/",
    #     video_units=30,
    #     size_list=1000,
    #     video={"pixels": 90, "aspect_ratio": [16, 9], "color": "GRAY"},
    # )
    # m = Unet2D(in_channels=30, channels=30, frames=30)

    # DL_DS = DataLoader(TD, batch_size=1, shuffle=True, num_workers=1)
    # for idx, batch in enumerate(DL_DS):
    #     train_features, train_labels = batch
    #     print(train_labels, train_features.shape)

    #     bottleneck, output = m(train_features)
    #     print(bottleneck.shape, output.shape)

    # train_features, train_labels = next(iter(TD))
    # print(train_labels, train_features.shape)
    # bottleneck, output = m(train_features.unsqueeze(0))
    # print(bottleneck.shape, output.shape)

    # def plot_mini_batch(imgs):
    #     img = imgs[0]
    #     print(img.shape)

    #     plt.figure(figsize=(20, 10))
    #     for i in range(img.shape[0]):
    #         plt.subplot(5, 6, i + 1)
    #         v_img = img[i].detach().numpy()
    #         # mask = masks[i, ...].permute(1, 2, 0).detach().numpy()
    #         plt.imshow(v_img)
    #         # plt.imshow(mask, alpha=0.5)

    #         plt.axis("Off")
    #     plt.tight_layout()
    #     plt.show()

    # plot_mini_batch(output)
    # plot_mini_batch(bottleneck)
    # for i in range(1, cols * rows + 1):
    #     figure.add_subplot(rows, cols, i)
    #     plt.axis("off")
    #     plt.imshow(bottleneck[0][i - 1], cmap="gray")
    # plt.title(train_labels)
    # plt.show()

    # Fin App
    log.info("End main program loop ...")
    log.info("Stop App ...")
