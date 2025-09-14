import base64
from uu import Error
import requests
from typing import Dict
import cv2
from dataclasses import dataclass

# Note: These imports work when running from the multiprocessing directory
try:
    from .db.db_logger import OAIX_db_Logger, LoggerLevel
    from .app_base import AppBase
except ImportError:
    # Fallback for when running directly
    from db.db_logger import OAIX_db_Logger, LoggerLevel
    from app_base import AppBase

@dataclass
class WebhookFrame():
    cameraId:str
    lot:str
    expiry:str
    all_text:str
    mime:str
    imageBase64:str

class Webhook(AppBase):
    def to_base64(self, frame, include_data_url=True):
        _, buffer = cv2.imencode('.jpg', frame)
        frame_based64 = base64.b64encode(buffer).decode('utf-8')
        if include_data_url:
            return f"data:image/jpeg;base64,{frame_based64}"
        
        return frame_based64

    def resize_frame(self, frame):
        height, width, _ = frame.shape
        half_h = height // 2
        half_w = width // 2
        return cv2.resize(frame, (half_w, half_h), cv2.INTER_LINEAR)
    
    def send_webhook(self, webhook_url: str, webhook_frame: WebhookFrame) -> bool:
        
        body = {
            "cameraId":webhook_frame.cameraId,
            "lot":webhook_frame.lot,
            "expiry":webhook_frame.expiry,
            "mime":webhook_frame.mime,
            "all_text":webhook_frame.all_text,
            "imageBase64":webhook_frame.imageBase64
        }

        headers = {
            "Content-Type":"application/json"
        }
        print(f"Webhook:{webhook_url=}")
        response = None

        try:
            response = requests.post(webhook_url, json=body, headers=headers, timeout=10)
            print(response.status_code)
            print(response.text) 
        except requests.exceptions.RequestException as e:
            print(e)
            err = f"Webhook:post request failed"
            print(err)
            self.db_logit(err, LoggerLevel.ERROR)
            return False

        success:bool = True if response and response.status_code == 200 else False
        return success