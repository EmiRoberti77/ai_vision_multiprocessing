from enum import Enum
import os
import cv2
import time
import json
import signal
import threading
from typing import List, Dict, Tuple
from dataclasses import dataclass
from webhook import Webhook, WebhookFrame

import torch
from ultralytics import YOLO
from ocr_processor import OCRProcessor


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"{ROOT=}")

class Detection_processor_type(Enum):
    ANY = 1
    CPU = 2
    GPU = 3

@dataclass
class DetectionParams:
    name:str
    video_source:str
    webhook_call_back_url:str
    rotate_90_clock:bool
    processor_type:Detection_processor_type


class Detection():
    def __init__(self, detection_params:DetectionParams) -> None:
        self.name = detection_params.name
        self.video_source = detection_params.video_source
        self.webhook_callback = detection_params.webhook_call_back_url
        self.rotate_90_clock = detection_params.rotate_90_clock
        self.processor_type = detection_params.processor_type
        self._stop_event = threading.Event()
        self.webhook = Webhook()
        self.running = False

    def create_annotated_frame(self, model: YOLO, frame, dets, ocr_results, rotate_for_ocr: bool = False):
        annotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) if rotate_for_ocr else frame.copy()

        # Draw YOLO boxes only on non-rotated frame
        if not rotate_for_ocr:
            for bbox, cls_id, conf in dets:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if hasattr(model, 'names') and cls_id in getattr(model, 'names', {}):
                    label = model.names[cls_id]
                else:
                    label = str(cls_id)
                cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 2)

        # Draw OCR results (handles both rotated and non-rotated)
        annotated = OCRProcessor.draw_ocr_results(OCRProcessor, annotated, ocr_results) if False else annotated
        # Use instance method instead when available in worker
        return annotated


    def save_outputs(self, channel_name: str, channel_run: str, frame, dets, ocr_results,
        model: YOLO, ocr: OCRProcessor):
        timestamp = int(time.time())
        save_dir = os.path.join(ROOT, channel_name, channel_run)
        os.makedirs(save_dir, exist_ok=True)

        # Normal annotated frame
        normal_frame = frame.copy()
        for bbox, cls_id, conf in dets:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(normal_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = model.names[cls_id] if hasattr(model, 'names') else str(cls_id)
            cv2.putText(normal_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 2)
        normal_frame = ocr.draw_ocr_results(normal_frame, ocr_results)
        normal_path = os.path.join(save_dir, f"{channel_name}_normal_{timestamp}.jpg")
        cv2.imwrite(normal_path, normal_frame)

        # Build JSON-safe OCR payload (avoid numpy types)
        safe_lines = []
        for line in ocr_results.get('lines', []):
            bbox = line.get('bbox', [])
            # bbox may be list of lists or numpy arrays; coerce to ints
            try:
                bbox_py = [[int(pt[0]), int(pt[1])] for pt in bbox]
            except Exception:
                bbox_py = []
            safe_lines.append({
                "text": str(line.get('text', '')),
                "confidence": float(line.get('confidence', 0.0)),
                "bbox": bbox_py
            })

        json_payload = {
            "timestamp": int(timestamp),
            "channel": str(channel_name),
            "yolo_detections": int(len(dets)),
            "ocr_results": {
                "text_count": int(ocr_results.get('text_count', 0) or 0),
                "lot": str(ocr_results.get('lot', '')),
                "expiry": str(ocr_results.get('expiry', '')),
                "full_text": str(ocr_results.get('text', '')),
                "lines": safe_lines
            }
        }
        with open(os.path.join(save_dir, f"{channel_name}_normal_{timestamp}.json"), 'w') as f:
            json.dump(json_payload, f, indent=2)

        # Rotated OCR frame only if any text was found
        if ocr_results.get('text_count', 0) > 0:
            rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            rotated = ocr.draw_ocr_results(rotated, ocr_results)
            rotated_path = os.path.join(save_dir, f"{channel_name}_ocr_{timestamp}.jpg")
            cv2.imwrite(rotated_path, rotated)
            with open(os.path.join(save_dir, f"{channel_name}_ocr_{timestamp}.json"), 'w') as f:
                json.dump({**json_payload, "frame_type": "ocr_rotated"}, f, indent=2)


    def start_worker(self, model_path: str = 'yolo11m.pt'):
        # Reduce OpenCV logs
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except Exception:
            pass

        print(f"[{self.name}] Starting worker for {self.video_source}")

        # Video
        cap = cv2.VideoCapture(self.video_source)
        using_fallback = False
        if not cap.isOpened():
            fallback = os.path.join(ROOT, 'rtsp_streamer', 'videos', 'emi_test.mp4')
            print(f"[{self.name}] RTSP failed, using fallback: {fallback}")
            cap = cv2.VideoCapture(fallback)
            using_fallback = cap.isOpened()
            if not using_fallback:
                print(f"[{self.name}] ERROR: Fallback not available. Exiting worker.")
                return

        # Device
        use_cuda = False
        try:
            use_cuda = torch.cuda.is_available()
        except Exception:
            use_cuda = False
        device = 'cuda' if use_cuda else 'cpu'

        # Models
        model = YOLO(model_path)
        try:
            model.to(device)
        except Exception:
            device = 'cpu'
        ocr = OCRProcessor(languages=['en'], gpu=(device == 'cuda'))

        channel_run = f"run-{int(time.time())}"
        self.running = True
        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    if using_fallback:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                    if not ok:
                        time.sleep(0.05)
                        continue

                t0 = time.time()
                try:
                    res = model(frame, imgsz=640, conf=0.25, verbose=False, device=device)
                except Exception:
                    # Last resort: force CPU
                    res = model(frame, imgsz=640, conf=0.25, verbose=False, device='cpu')
                    device = 'cpu'
                yolo_time = (time.time() - t0) * 1000

                # Parse detections
                dets: List[Tuple[List[int], int, float]] = []
                for r in res:
                    if getattr(r, 'boxes', None) is None:
                        continue
                    for b in r.boxes:
                        dets.append((
                            [int(round(x)) for x in b.xyxy[0].tolist()],
                            int(b.cls.item()),
                            float(b.conf.item())
                        ))

                t1 = time.time()
                ocr_results: Dict = ocr.process_frame(frame, rotate_iphone=self.rotate_90_clock)
                ocr_time = (time.time() - t1) * 1000

                self.save_outputs(self.name, channel_run, frame, dets, ocr_results, model, ocr)

                if dets or ocr_results.get('text_count', 0) > 0:
                    webhook_frame = WebhookFrame(
                        cameraId=self.name,
                        lot=ocr_results.get('lot'),
                        expiry=ocr_results.get('expiry'),
                        mime='detecton_data',
                        imageBase64=self.webhook.to_base64(frame=self.webhook.resize_frame(frame), include_data_url=True)
                    )   
                    if self.webhook.send_webhook(self.webhook_callback, webhook_frame):
                        print(f"Success frame sent")
                    
                    print(f"[{self.name}] YOLO={len(dets)} OCR={ocr_results.get('text_count', 0)} | YOLO={yolo_time:.1f}ms OCR={ocr_time:.1f}ms")

                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            self.running = False
            print(f"[{self.name}] Exiting worker")

    
    def stop_worker(self):
        print(f"Thread_stop:{self.name}")
        self._stop_event.set()
        return self._stop_event.is_set()



class DetectionManager():
    def __init__(self):
        self._lock = threading.Lock()
        self.detections:Dict[str, Detection] = {}
        self.threads:Dict[str, threading.Thread] = {}

    def add(self, detection_params:DetectionParams)->bool:
        with self._lock:
            if detection_params.name not in self.detections:
                d = Detection(detection_params=detection_params)
                t = threading.Thread(
                    target=d.start_worker,
                    args=(),
                    daemon=True
                )
                self.detections[detection_params.name] = d
                self.threads[detection_params.name] = t
                return True

        return False

    
    def start(self, name:str)->bool:
        with self._lock:
            if name in self.detections and name in self.threads:
                t = self.threads[name]
                t.start()
                return True
        
        return False

    
    def stop(self, name:str)->bool:
        with self._lock: 
            if name in self.detections and name in self.threads:
                if self.detections[name].stop_worker():
                    print(f"DM:stop_worker:{name}")    
                    self.threads[name].join()
                    print(f"DM:thread_join:{name}")
                    return True
        
        return False

    
    def remove(self, name:str)->bool:
        with self._lock:
            if name in self.detections and name in self.threads:
                del self.detections[name]
                del self.threads[name]
                print(f"DM:removed:{name}")
                return True
        
        return False


def process():
    urls = [
        'rtsp://172.23.23.15:8554/mystream_4',
        'rtsp://172.23.23.15:8554/mystream_3'
    ]

    dm = DetectionManager()
    dm.add('channel1','rtsp://172.23.23.15:8554/mystream_4')
    dm.start('channel1')

    time.sleep(20)

    dm.stop("channel1")


# process()



