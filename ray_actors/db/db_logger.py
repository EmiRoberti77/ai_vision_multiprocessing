from enum import IntEnum

from torch import ErrorReport
from .db_manager import SessionLocal, AppLogger
from .error_codes import ErrorCode, ErrorSeverity, format_error_message
from .error_codes import get_error_info
from .info_codes import InfoCode
import threading

class LoggerLevel(IntEnum):
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class OAIX_db_Logger():
    def __init__(self) -> None:
        self.lock = threading.Lock()

    def app_logger(self, log_code:ErrorCode | InfoCode,  msg:str, level:LoggerLevel=LoggerLevel.INFO):
        with self.lock:
            session = SessionLocal()
            try:
                log_msg = AppLogger(
                    level=level,
                    log_code=log_code,
                    message=msg
                ) 
                session.add(log_msg)
                session.commit()
            except:
                print('OAIX_Logger:error')
            finally:
                session.close()    

    
    def log_error(self, error_code: ErrorCode , context: str = "", details: str = ""):
        error_info = get_error_info(error_code)
        message = format_error_message(error_code, context, details)
        
        # Map ErrorSeverity to LoggerLevel
        severity_map = {
            ErrorSeverity.INFO: LoggerLevel.INFO,
            ErrorSeverity.WARNING: LoggerLevel.WARNING,
            ErrorSeverity.ERROR: LoggerLevel.ERROR,
            ErrorSeverity.CRITICAL: LoggerLevel.CRITICAL
        }
        
        log_level = severity_map.get(error_info['severity'], LoggerLevel.ERROR)
        self.app_logger(error_code, message, log_level)

