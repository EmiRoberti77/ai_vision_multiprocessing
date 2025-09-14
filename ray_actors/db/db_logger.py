from enum import IntEnum
from .db_manager import SessionLocal, AppLogger
import threading

class LoggerLevel(IntEnum):
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class OAIX_db_Logger():
    def __init__(self) -> None:
        self.lock = threading.Lock()

    def app_logger(self, msg:str, level:LoggerLevel=LoggerLevel.INFO):
        with self.lock:
            session = SessionLocal()
            try:
                log_msg = AppLogger(
                    level=level,
                    message=msg
                ) 
                session.add(log_msg)
                session.commit()
            except:
                print('OAIX_Logger:error')
            finally:
                session.close()

