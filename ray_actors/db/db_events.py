from db.db_manager import SessionLocal, WebhookEvent
import threading
import os
import sys

class OAIX_db_Event():
    def __init__(self) -> None:
        self.lock = threading.Lock()

    def app_ocr_event(self, camera_name:str, all_text:str, lot:str=None, expiry:str=None, image_path:str=None, mime:str='ocr_data'):
        with self.lock:
            session = SessionLocal()
            try:
                ocr_event = WebhookEvent(
                    camera_name=camera_name,
                    all_text=all_text,
                    lot=lot,
                    expiry=expiry,
                    image_path=image_path,
                    mime=mime
                ) 
                session.add(ocr_event)
                session.commit()
            except:
                print('OAIX_db_ocr_event:error')
            finally:
                session.close()