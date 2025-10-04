from enum import Enum
import os
import cv2
import time
import json
import sys
import threading
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from app_base import AppBase
from db.db_events import OAIX_db_Event
from webhook import Webhook, WebhookFrame
import numpy as np

import torch
from ultralytics import YOLO
from ocr_processor import OCRProcessor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if ROOT not in sys.path:
    print(f'ROOT:append:{ROOT=}')
    sys.path.append(ROOT)

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
    model_name:str


class Detection():
    def __init__(self, detection_params:DetectionParams) -> None:
        self.name = detection_params.name
        self.video_source = detection_params.video_source
        self.webhook_callback = detection_params.webhook_call_back_url
        self.rotate_90_clock = detection_params.rotate_90_clock
        self.processor_type = detection_params.processor_type
        self.model_path = detection_params.model_name
        self._stop_event = threading.Event()
        self.webhook = Webhook()
        self.running = False
        # OCR gating: run OCR only when a detection is stable across frames
        self.min_stable_frames = 5
        self.iou_threshold = 0.6
        self.min_area_ratio = 0.03  # require bbox to be at least 3% of frame area (tighter)
        self.focus_laplacian_thresh = 120.0  # require ROI sharpness above this
        self.ocr_cooldown_frames = 30  # run OCR at most once every N stable frames
        self._cooldown_remaining = 0
        self._last_roi_hash: Optional[int] = None
        self._last_text_signature: Optional[str] = None
        self._stable_target = None  # { 'bbox': [x1,y1,x2,y2], 'cls_id': int, 'count': int }
        self.print_start_settings()


    def print_start_settings(self):
        print(f"[{self.name}] Startup settings:")
        try:
            proc_type = self.processor_type.name if isinstance(self.processor_type, Detection_processor_type) else str(self.processor_type)
        except Exception:
            proc_type = str(self.processor_type)
        settings = {
            "video_source": self.video_source,
            "webhook_callback": self.webhook_callback,
            "rotate_90_clock": self.rotate_90_clock,
            "processor_type": proc_type,
            "model_path": self.model_path,
            "min_stable_frames": self.min_stable_frames,
            "iou_threshold": self.iou_threshold,
            "min_area_ratio": self.min_area_ratio,
            "focus_laplacian_thresh": self.focus_laplacian_thresh,
            "ocr_cooldown_frames": self.ocr_cooldown_frames,
        }
        for k, v in settings.items():
            print(f"  {k}: {v}")

    def _bbox_area(self, bbox):
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = self._bbox_area(a)
        area_b = self._bbox_area(b)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _update_stability(self, dets, frame_shape) -> bool:
        """
        Track the most confident detection and check if it stays in place
        for at least self.min_stable_frames with IoU >= self.iou_threshold
        and minimum area.
        Returns True when stable, else False.
        """
        if not dets:
            self._stable_target = None
            self._cooldown_remaining = 0
            return False

        # Choose the highest confidence detection
        best_det = max(dets, key=lambda d: d[2])  # (bbox, cls_id, conf)
        bbox, cls_id, conf = best_det

        # Filter by area to approximate "in focus"
        h, w = frame_shape[:2]
        if self._bbox_area(bbox) < self.min_area_ratio * (w * h):
            self._stable_target = None
            self._cooldown_remaining = 0
            return False

        if self._stable_target and self._stable_target.get('cls_id') == cls_id:
            iou = self._iou(self._stable_target['bbox'], bbox)
            if iou >= self.iou_threshold:
                self._stable_target['count'] += 1
                self._stable_target['bbox'] = bbox
            else:
                self._stable_target = { 'bbox': bbox, 'cls_id': cls_id, 'count': 1 }
                self._cooldown_remaining = 0
        else:
            self._stable_target = { 'bbox': bbox, 'cls_id': cls_id, 'count': 1 }
            self._cooldown_remaining = 0

        return self._stable_target['count'] >= self.min_stable_frames

    def _variance_of_laplacian(self, img: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = img
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _crop_expand(self, frame: np.ndarray, bbox, margin_ratio: float = 0.25) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        mx = int(bw * margin_ratio)
        my = int(bh * margin_ratio)
        cx1 = max(0, x1 - mx)
        cy1 = max(0, y1 - my)
        cx2 = min(w, x2 + mx)
        cy2 = min(h, y2 + my)
        return frame[cy1:cy2, cx1:cx2]

    def _ahash(self, img: np.ndarray, size: int = 8) -> int:
        if img is None or img.size == 0:
            return 0
        try:
            small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (size, size))
        except Exception:
            small = cv2.resize(img, (size, size))
        avg = small.mean()
        bits = (small > avg).flatten().astype(np.uint8)
        hash_val = 0
        for bit in bits:
            hash_val = (hash_val << 1) | int(bit)
        return int(hash_val)

    def _hamming_distance(self, a: int, b: int) -> int:
        return int(bin(a ^ b).count('1'))


    def save_outputs(self, channel_name: str, channel_run: str, frame, dets, ocr_results,
        model: YOLO, ocr):
        timestamp = int(time.time())
        save_dir = os.path.join(ROOT, channel_name, channel_run)
        os.makedirs(save_dir, exist_ok=True)

        # Normal annotated frame
        # normal_frame = frame.copy()
        # for bbox, cls_id, conf in dets:
        #     x1, y1, x2, y2 = bbox
        #     cv2.rectangle(normal_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     label = model.names[cls_id] if hasattr(model, 'names') else str(cls_id)
        #     cv2.putText(normal_frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 2)
        
        ocr_frame = ocr.draw_ocr_results(frame, ocr_results)
        ocr_path = os.path.join(save_dir, f"{channel_name}_ocr_{timestamp}.jpg")
        cv2.imwrite(ocr_path, ocr_frame)


    def start_worker(self):
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
        model = YOLO(self.model_path)
        try:
            model.to(device)
        except Exception:
            device = 'cpu'
        
        # Initialize OCR with runtime fallback if Paddle backend is present but not installed:
        ocr = OCRProcessor(channel_name=self.name, languages=['en'], gpu=(device == 'cuda'))
        

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

                # Gate OCR by stability across frames to avoid over-processing
                is_stable = self._update_stability(dets, frame.shape)
                # Manage cooldown
                if self._cooldown_remaining > 0:
                    self._cooldown_remaining -= 1
                do_ocr = is_stable and self._cooldown_remaining == 0

                ocr_triggered = False
                ocr_results: Dict = {
                    'text': '',
                    'lot': '',
                    'expiry': '',
                    'lines': [],
                    'processing_time_ms': 0.0,
                    'text_count': 0,
                    'device': getattr(ocr, 'device', 'cpu')
                }

                # Decide whether to OCR based on focus and duplicates on ROI
                if do_ocr and self._stable_target:
                    roi = self._crop_expand(frame, self._stable_target['bbox'], margin_ratio=0.3)
                    # Focus check
                    focus_val = self._variance_of_laplacian(roi)
                    # Duplicate ROI check using perceptual hash
                    roi_hash = self._ahash(roi)
                    is_duplicate_roi = (self._last_roi_hash is not None and self._hamming_distance(roi_hash, self._last_roi_hash) <= 2)

                    if focus_val >= self.focus_laplacian_thresh and not is_duplicate_roi:
                        t1 = time.time()
                        # Run OCR on the ROI to reduce load and improve focus
                        ocr_roi = ocr.process_frame(roi, rotate_90_clock=self.rotate_90_clock, save_frame=True)
                        ocr_results = ocr_roi
                        ocr_time = (time.time() - t1) * 1000
                        ocr_triggered = True
                        self._last_roi_hash = roi_hash
                        # Start cooldown regardless of OCR outcome to avoid hammering
                        self.save_outputs(self.name, channel_run, roi, dets, ocr_results, model, ocr)
                        self._cooldown_remaining = self.ocr_cooldown_frames
                    else:
                        ocr_time = 0.0
                        # Even if we skip OCR due to focus/duplicate, keep cooldown to prevent spamming
                        self._cooldown_remaining = max(self._cooldown_remaining, int(self.ocr_cooldown_frames / 2))
                else:
                    ocr_time = 0.0

                # Save annotated outputs and select best frame for webhook
                # self.save_outputs(self.name, channel_run, frame, dets, ocr_results, model, ocr)

                # Only send when OCR actually ran and produced some text, and text changed
                sent = False
                if ocr_triggered and ocr_results.get('text_count', 0) > 0:
                    text_sig = f"{ocr_results.get('lot','')}|{ocr_results.get('expiry','')}|{ocr_results.get('text','')[:64]}"
                    if self._last_text_signature != text_sig:
                        # Choose best image to send: prefer saved OCR frame if exists
                        img_b64 = self.webhook.to_base64(
                            frame=self.webhook.resize_frame(frame),
                            include_data_url=True
                        )
                        if ocr_results.get('ocr_image_path'):
                            try:
                                ocr_img = cv2.imread(ocr_results.get('ocr_image_path'))
                                if ocr_img is not None:
                                    img_b64 = self.webhook.to_base64(
                                        frame=self.webhook.resize_frame(ocr_img),
                                        include_data_url=True
                                    )
                            except Exception:
                                pass

                        webhook_frame = WebhookFrame(
                            cameraId=self.name,
                            lot=ocr_results.get('lot'),
                            expiry=ocr_results.get('expiry'),
                            all_text=ocr_results.get('text'),
                            mime='detecton_data',
                            imageBase64=img_b64
                        )
                        if self.webhook.send_webhook(self.webhook_callback, webhook_frame):
                            sent = True
                            self._last_text_signature = text_sig
                            print(f"Success frame sent")
                            # save event in the database
                            oaix_db_event = OAIX_db_Event()
                            oaix_db_event.app_ocr_event(
                                camera_name=self.name,
                                all_text=ocr_results.get('text'),
                                lot=ocr_results.get('lot'),
                                expiry=ocr_results.get('expiry'),
                                image_path=ocr_results.get('ocr_image_path'),
                                mime='OCR_EVENT'
                            )

                stable_cnt = self._stable_target['count'] if self._stable_target else 0
                # print(f"[{self.name}] YOLO={len(dets)} OCR={ocr_results.get('text_count', 0)} | YOLO={yolo_time:.1f}ms OCR={ocr_time:.1f}ms | stable={stable_cnt}/{self.min_stable_frames} | ocr={'Y' if ocr_triggered else 'N'} | sent={'Y' if sent else 'N'} | cd={self._cooldown_remaining}")

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



