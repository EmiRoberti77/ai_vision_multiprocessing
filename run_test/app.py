import os
import time
import threading
import hashlib
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from dotenv import load_dotenv

# ---- your modules ----
import utils
from OAIX_GOCR_Detection import OAIX_GOCR_Detection as OCR
# ----------------------

load_dotenv()

# ---------- CONFIG ----------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
SAVE_RUN_PATH = os.path.join(ROOT, "runs")
MODELS_FOLDER = os.path.join(ROOT, "models")
MODELS_LIST = ("oaix_medicine_v1.pt", "yolo11m.pt")
MODEL_PATH = os.path.join(ROOT, MODELS_FOLDER, MODELS_LIST[0])

RTSP_URL = os.getenv("RTSP_URL", "rtsp://172.23.23.15:8554/mystream_5")
RTSP_RECONNECT_DELAY = float(os.getenv("RTSP_RECONNECT_DELAY", "2.0"))  # seconds before retry
RTSP_WARMUP_READS = int(os.getenv("RTSP_WARMUP_READS", "5"))            # discard some frames on connect
FRAME_STALE_MS = int(os.getenv("FRAME_STALE_MS", "1500"))               # max allowed age of latest frame
READ_SLEEP_MS = int(os.getenv("READ_SLEEP_MS", "1"))                    # small sleep to yield CPU

MIN_CONF = float(os.getenv("YOLO_MIN_CONF", "0.30"))
IOU = float(os.getenv("YOLO_IOU", "0.40"))

LABEL_CLASS_ID = os.getenv("LABEL_CLASS_ID")
LABEL_CLASS_ID = int(LABEL_CLASS_ID) if LABEL_CLASS_ID not in (None, "", "None") else None
# ----------------------------


# ---------- YOLO wrapper ----------
class YoloDetect:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, frame):
        boxes, confs, clids = [], [], []
        results = self.model.predict(frame, iou=IOU, conf=MIN_CONF, verbose=False)
        for r in results:
            for b in r.boxes:
                boxes.append(list(map(int, b.xyxy[0].tolist())))
                confs.append(float(b.conf))
                clids.append(int(b.cls))
        return boxes, confs, clids

    @staticmethod
    def crop(frame, x1, y1, x2, y2):
        if x2 > x1 and y2 > y1:
            return frame[y1:y2, x1:x2].copy()
        return None
# ----------------------------------


# ---------- Persistent RTSP reader ----------
class RTSPReader:
    """
    Background thread that always reads RTSP and holds the latest frame in memory.
    Endpoint fetches the latest frame via get_latest().
    Auto-reconnects on errors.
    """
    def __init__(self, url: str):
        self.url = url
        self._lock = threading.Lock()
        self._cap: Optional[cv2.VideoCapture] = None
        self._latest: Optional[Tuple[float, any]] = None  # (timestamp_ms, frame)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rtsp-reader", daemon=True)

    def start(self):
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass

    def _open(self) -> bool:
        self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            return False
        # reduce buffering if backend supports it
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        # warmup
        for _ in range(max(0, RTSP_WARMUP_READS)):
            self._cap.read()
        return True

    def _run(self):
        while not self._stop.is_set():
            try:
                if not self._open():
                    time.sleep(RTSP_RECONNECT_DELAY)
                    continue

                while not self._stop.is_set():
                    ok, frame = self._cap.read()
                    if not ok or frame is None:
                        # lost connection; break to reopen
                        break

                    ts_ms = time.time() * 1000.0
                    with self._lock:
                        # print(f"=>{ts_ms}")
                        self._latest = (ts_ms, frame)

                    # tiny sleep to avoid pegging CPU if camera runs very high FPS
                    time.sleep(READ_SLEEP_MS)
                    # print(f"RTSP_THREAD {time.time()}")

            except Exception:
                # swallow and retry
                pass
            finally:
                try:
                    if self._cap is not None:
                        self._cap.release()
                except Exception:
                    pass
                self._cap = None

            # brief delay before reconnect
            time.sleep(RTSP_RECONNECT_DELAY)

    def get_latest(self) -> Optional[any]:
        with self._lock:
            if self._latest is None:
                return None
            ts_ms, frame = self._latest
            # print(f"<={ts_ms}")
            # return a copy to avoid concurrent mutations
            return frame.copy()
# -------------------------------------------


def pick_detection(boxes, confs, clids):
    if not boxes:
        return None
    if LABEL_CLASS_ID is not None:
        candidates = [(i, c) for i, (cls, c) in enumerate(zip(clids, confs)) if cls == LABEL_CLASS_ID]
        if candidates:
            return max(candidates, key=lambda x: x[1])[0]
    return int(max(range(len(boxes)), key=lambda i: confs[i]))


# ---------- App wiring ----------
app = FastAPI(title="Medicine Label Triggered Processor (Persistent RTSP)")
reader = RTSPReader(RTSP_URL)
reader.start()

detector = YoloDetect(MODEL_PATH)
ocr = OCR()

# Track processed frames to avoid duplicates
processed_frames = set()
last_frame_hash = None


@app.get("/health")
def health():
    frame = reader.get_latest()
    return {
        "ok": True,
        "rtsp_url": RTSP_URL,
        "frame_available": frame is not None,
        "model_loaded": True,
    }


@app.get("/stats")
def stats():
    """Get frame processing statistics"""
    return {
        "total_processed_frames": len(processed_frames),
        "last_frame_hash": last_frame_hash,
        "rtsp_url": RTSP_URL,
        "frame_stale_ms": FRAME_STALE_MS,
    }


@app.post("/process")
def process(save_artifacts: bool = Query(default=True, description="Save ROI/full/final images under /runs")):
    global last_frame_hash, processed_frames
    
    # 1) Get latest frame (instant)
    import datetime
    current = utils.file_name('current')
    frame = reader.get_latest()
    if frame is None:
        raise HTTPException(status_code=503, detail="No fresh frame available (stream not ready or stale)")
    # 2) YOLO detection
    cv2.imwrite(current, frame)
    print(f"{frame.shape[:2]},{frame.nbytes}")
    boxes, confs, clids = detector.predict(frame)
    if not boxes:
        return JSONResponse(
            status_code=200,
            content={"message": "No detections", "detections": 0, "result": None},
        )

    idx = pick_detection(boxes, confs, clids)
    x1, y1, x2, y2 = boxes[idx]
    roi = YoloDetect.crop(frame, x1, y1, x2, y2)
    if roi is None or roi.size == 0:
        return JSONResponse(
            status_code=200,
            content={"message": "Detection had empty ROI", "detections": len(boxes), "result": None},
        )

    # 4) Save artifacts (optional)
    run_folder = utils.create_run_folder_output(SAVE_RUN_PATH, "run") if save_artifacts else None
    crop_path = full_path = final_path = None

    # if save_artifacts:
    #     annotated = frame.copy()
    #     cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
    #     cv2.putText(
    #         annotated,
    #         f"{confs[idx]:.2f}:{clids[idx]}",
    #         (x1, y1 - 10),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.7,
    #         (0, 255, 0),
    #         2,
    #         cv2.LINE_AA,
    #     )

    full_path = os.path.join(run_folder, utils.file_name("med_full"))
    cv2.imwrite(full_path, frame)

    crop_path = os.path.join(run_folder, utils.file_name("med_roi"))
    cv2.imwrite(crop_path, roi)

    # tmp_folder = utils.create_run_folder_output(SAVE_RUN_PATH, "tmp")
    # crop_path = os.path.join(tmp_folder, "roi.jpg")
    # cv2.imwrite(crop_path, roi)

    result = ocr.gem_detect(crop_path)

    if save_artifacts:
        final = roi.copy()
        cv2.putText(final, f"lot:{result.get('lot_number')}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(final, f"exp:{result.get('expiry_date')}", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        final_path = os.path.join(run_folder, utils.file_name("final"))
        cv2.imwrite(final_path, final)

    payload: Dict[str, Any] = {
        "message": "Processed latest frame",
        "detections": len(boxes),
        "chosen_detection": {
            "box": [x1, y1, x2, y2],
            "confidence": round(confs[idx], 3),
            "class_id": int(clids[idx]),
        },
        "result": {
            "lot_number": result.get("lot_number"),
            "expiry_date": result.get("expiry_date"),
        },
        "artifacts": {
            "full_frame_path": full_path,
            "crop_path": crop_path,
            "final_path": final_path,
        } if save_artifacts else None,
    }
    return JSONResponse(status_code=200, content=payload)
# -------------------------------------------


# Optional: graceful shutdown hook (uvicorn handles signals; keeping explicit stop for completeness)
@app.on_event("shutdown")
def on_shutdown():
    reader.stop()
