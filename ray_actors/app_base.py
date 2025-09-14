from db.db_logger import OAIX_db_Logger, LoggerLevel

class AppBase():
    def __init__(self) -> None:
        self.app_logger = OAIX_db_Logger()

    def db_logit(self, msg:str, level:LoggerLevel=LoggerLevel.INFO):
        self.app_logger.app_logger(msg, level)