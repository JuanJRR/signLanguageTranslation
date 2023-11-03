import pandas as pd

from utils.logger import Logger


class FilesCSV:
    """Utilities for manipulating CSV files"""

    @staticmethod
    def read_csv(annotations_file: str) -> tuple:
        """Permite cargar un archivo CSV, que contenga la estructura [ID, FIRMA]

        :param annotations_file: CSV file path to upload
        :type annotations_file: str
        :return: A tuple containing [ID, SIGNATURE]
        :rtype: tuple
        """
        # settings
        logger = Logger()
        log = logger.config_logging()
        log.debug("Starting file upload ...")

        # Loading information
        metadata_file = pd.read_csv(annotations_file, sep=",")
        id_file = metadata_file["ID"].to_numpy()
        label_file = metadata_file["FIRMA"].to_numpy()

        del metadata_file

        log.debug("File upload completed")
        log.info("File uploaded")

        return id_file, label_file
