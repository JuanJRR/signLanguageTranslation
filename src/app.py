import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from dataWrangling.datasetWordSLC import DatasetWordSLC
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

    TD = DatasetWordSLC(
        annotations_file=False,
        annotations_dir="data/raw/10_words_3_people/000_10_words_3_people.csv",
        items_dir="data/raw/10_words_3_people/",
        video_units=30,
        size_list=1000,
        video={"pixels": 90, "aspect_ratio": [16, 9], "color": "GRAY"},
    )

    TD_1 = DatasetWordSLC(
        annotations_file=False,
        annotations_dir="data/raw/10_words_3_people/000_10_words_3_people.csv",
        items_dir="data/raw/10_words_3_people/",
        video_units=30,
        size_list=1000,
        video={"pixels": 90, "aspect_ratio": [16, 9], "color": "GRAY"},
    )

    DL_DS = DataLoader(TD, batch_size=32, shuffle=True, num_workers=2)
    for idx, batch in enumerate(DL_DS):
        # Print the 'text' data of the batch
        print(idx)
        # Print the 'class' data of batch
        train_features, train_labels = batch
        print(train_labels)
        print(train_features.shape)

    # figure = plt.figure()
    # cols, rows = 5, 4

    # for i in range(1, cols * rows + 1):
    #     figure.add_subplot(rows, cols, i)
    #     plt.axis("off")
    #     plt.imshow(train_features[i - 1], cmap="gray")
    # plt.title(train_labels)
    # plt.show()

    # Fin App
    log.info("End main program loop ...")
    log.info("Stop App ...")
