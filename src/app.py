import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from torch import nn
from torch.utils.data import DataLoader

from data.dataListGenc import DataListGeneratorClassifier
from dataWrangling.datasetWordSLC import DatasetWordSLC
from models.unet_2d.model.unet2d import Unet2D
from models.unet_2d.optimisers.optimUnet2dSGD import OptimUnet2dSGD
from models.unet_2d.train.estimate_reasonable_lr import EstimateReasonableLr
from utils.logger import Logger

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
        seed = 1
        np.random.seed(seed)
        torch.manual_seed(seed)
        log.info("Defined seed: {}".format(seed))
    else:
        log.info("Random seed")

    # Execution App
    log.warning("Starting main program loop ...")

    # gen_c = DataListGeneratorClassifier(
    #     annotations_file="/media/juan/Archivos/Proyectos/signLanguageTranslation/data/raw/10_words_3_people/000_10_words_3_people.csv"
    # )
    # list, index = gen_c.generator(size_list=3)
    # gen_c.save_data_list(
    #     df_generator=list,
    #     save_dir="/media/juan/Archivos/Proyectos/signLanguageTranslation/data/processed/10SLC.csv",
    # )

    # aa = torch.randn(32, 1, 128, 256)
    # print(aa.shape)
    # a_max_0 = torch.argmax(aa, dim=1)
    # print(a_max_0.shape)

    # bb = torch.randn(2, 30, 128, 256)
    # acc = (a_max_0 == aa).sum()
    # print(acc)
    # a_max_1 = torch.argmax(aa, dim=1)
    # print(aa[0], aa[1])
    # print(a_max_0, a_max_1)

    # print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

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

    TD = DatasetWordSLC(
        use_annotation_list=False,
        ram_preload=True,
        annotations_dir="/media/juan/Archivos/Proyectos/signLanguageTranslation/data/processed/10SLC.csv",
        items_dir="data/raw/10_words_3_people/",
        video_units=30,
        size_list=10,
        video={"pixels": 128, "aspect_ratio": [16, 9], "color": "GRAY"},
    )

    # TD_1 = DatasetWordSLC(
    #     annotations_file=False,
    #     annotations_dir="data/raw/10_words_3_people/000_10_words_3_people.csv",
    #     items_dir="data/raw/10_words_3_people/",
    #     video_units=30,
    #     size_list=1000,
    #     video={"pixels": 90, "aspect_ratio": [16, 9], "color": "GRAY"},
    # )
    m = Unet2D(in_channels=30, channels=30, frames=30)
    optim = OptimUnet2dSGD.optim_sgd_1(model=m, lr=1e-6)

    DL_DS = DataLoader(TD, batch_size=1, shuffle=True)

    lrs, losses, accuracies = EstimateReasonableLr.estimate_lr(
        model=m, data_loader=DL_DS, optim=optim, max_lr=10, min_lr=1e-6
    )

    f1, ax1 = plt.subplots(figsize=(20, 10))
    ax1.plot(lrs, losses, label="lr")
    # ax1.plot(lrs, accuracies, label="acc")
    ax1.set_xscale("log")
    ax1.legend(loc="upper left")
    ax1.grid()
    plt.show()

    # lr = [1, 2, 3, 4, 5]
    # loss = [1, 2, 3, 4, 5]
    # coss = [5, 4, 3, 2, 1]
    # f1, ax1 = plt.subplots(figsize=(20, 10))
    # ax1.plot(lr, loss)
    # ax1.plot(lr, coss)
    # plt.show()

    # print(len(DL_DS))
    # for idx, batch in enumerate(DL_DS):
    #     print("for DL_DS: {0}".format(idx))
    #     train_features, train_labels = batch
    #     print(train_labels, train_features.shape)

    #     bottleneck, output = m(train_features)
    #     print(bottleneck.shape, output.shape)

    # print(train_labels, train_features.shape)
    # bottleneck, output = m(train_features.unsqueeze(0))
    # print(bottleneck.shape, output.shape)

    # def plot_mini_batch(imgs):
    #     img = imgs
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

    # for i in [0, 2, 6, 23, 304, 7]:
    #     train_features, train_labels = TD[i]

    #     plot_mini_batch(train_features)
    # plot_mini_batch(output)
    # plot_mini_batch(bottleneck)
    # for i in range(1, cols * rows + 1):
    #     figure.add_subplot(rows, cols, i)
    #     plt.axis("off")
    #     plt.imshow(bottleneck[0][i - 1], cmap="gray")
    # plt.title(train_labels)
    # plt.show()

    # Fin App
    log.warning("End main program loop ...")
    log.critical("Stop App ...")
