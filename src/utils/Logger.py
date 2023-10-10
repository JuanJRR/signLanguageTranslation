import logging
import logging.config

import yaml


class Logger:
    """ "Defining the event logger

    :return: Object logging.Logger
    :rtype: logging.Logger
    """

    def __init__(self, level_Log: str, setting_log: str) -> None:
        """Initialization

        :param level_Log: Logger level. DEV for development, TEST for evaluation, PROD for production
        :type level_Log: str
        :param setting_log: Location of logging.yaml configuration file
        :type setting_log: str
        """

        self.level = level_Log

        with open(setting_log, "r") as f:
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
