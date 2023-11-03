import numpy as np
import pandas as pd

from utils.filesCSV import FilesCSV
from utils.logger import Logger


class DataListGeneratorClassifier:
    """Generates a set of elements with discrete distribution from a set of elements.
    The set of elements used as seed must be in CSV format and contain the following columns:
    ID: File name and its extension, FIRMA: file label"""

    def __init__(self, annotations_file: str) -> None:
        """Builder

        :param annotations_file: Path to the original set of elements in CSV format
        :type annotations_file: str
        """

        # settings
        logger = Logger()
        self.log = logger.config_logging()

        # upload information
        id_file, label_file = FilesCSV.read_csv(annotations_file)

        self.data_file = {}
        self.data_size = len(id_file)

        for i in range(self.data_size):
            self.data_file[i] = {"ID": id_file[i], "FIRMA": label_file[i]}

        del id_file, label_file
        self.log.info("Built data list generator")

    def generator(self, size_list: int = 255) -> pd.DataFrame:
        """Generates a set of random elements that satisfy a discrete distribution
        from a list of elements used as a seed

        :param size_list: Generator final length, defaults to 255
        :type size_list: int, optional
        :return: Dataframe type listing, which complies with the ID, FIRMA structure
        :rtype: pd.DataFrame
        """
        self.log.debug("Starting data generator ...")

        list_generator = {}
        keys = np.random.randint(self.data_size, size=size_list)

        for i, item in enumerate(keys):
            list_generator[i] = self.data_file[item]

        data_list = []
        for i in range(len(list_generator)):
            data_list.append(list_generator[i])

        df_generator = pd.DataFrame(data_list)

        del keys, list_generator, data_list

        self.log.debug("Completed data generator")
        self.log.info("New set of elements created")

        return df_generator

    def save_data_list(self, df_generator: pd.DataFrame, save_dir: str) -> None:
        """Saves in a Dataframe type object in a CSV extension file.

        :param df_generator: Dataframe type object that complies with the ID, FIRMA structure.
        :type df_generator: pd.DataFrame
        :param save_dir: Path where to save the CSV file
        :type save_dir: str
        """
        df_generator.to_csv(save_dir, index=False)

        del df_generator
        self.log.info("Saved set of elements")
