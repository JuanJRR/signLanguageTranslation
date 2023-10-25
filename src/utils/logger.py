import logging
import logging.config
import os

import yaml


class Logger:
    """ "Defining the event logger

    :return: Object logging.Logger
    :rtype: logging.Logger
    """

    def __init__(self) -> None:
        """Initialization"""

        self.setting_log = str(os.getenv("setting_logging"))
        # self.setting_log = "../signLanguageTranslation/SettingLogging.yaml"
        self.level = os.getenv("level_Logging")

        with open(self.setting_log, "r") as f:
            self.log_cfg = yaml.safe_load(f.read())

    def config_logging(self) -> logging.Logger:
        """Defines how the logging will behave.

        :return: Object with logger configuration
        :rtype: logging.Logger
        """

        logging.config.dictConfig(self.log_cfg)

        if self.level == "DEV":
            log = logging.getLogger("DEV")
            log.setLevel(logging.DEBUG)
            return log
        elif self.level == "TEST":
            log = logging.getLogger("TEST")
            log.setLevel(logging.INFO)
            return log
        else:
            log = logging.getLogger("PROD")
            log.setLevel(logging.WARNING)
            return log
