from db.db_logger import OAIX_db_Logger, LoggerLevel
from db.db_events import OAIX_db_Event, WebhookEvent


class AppBase():
    def __init__(self) -> None:
        print('APP_BASE_INIT')
        self.app_logger = OAIX_db_Logger()
        self.app_event = OAIX_db_Event()

    def db_logit(self, msg:str, level:LoggerLevel=LoggerLevel.INFO)->None:
        self.app_logger.app_logger(msg, level)
    
    def db_detection_event(self,  camera_name:str, all_text:str, lot:str=None, expiry:str=None, image_path:str=None, mime:str='ocr_data')->None:
        self.app_event.app_ocr_event(
            camera_name=camera_name,
            all_text=all_text,
            lot=lot,
            expiry=expiry,
            image_path=image_path,
            mime=mime
        )