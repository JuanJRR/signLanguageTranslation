from utils.Logger import Logger

if __name__ == "__main__":
    logger = Logger(
        level_Log="DEV",
        setting_log="../signLanguageTranslation/SettingLogging.yaml",
    )
    log = logger.config_logging()
