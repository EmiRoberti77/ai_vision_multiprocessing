import os
import time
import threading
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from ultralytics import YOLO
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from db.og_lotes.readlotes import LoteDB
import utils
from OAIX_GOCR_Detection import OAIX_GOCR_Detection as OCR
load_dotenv()

# ---------- CONFIG ----------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
SAVE_RUN_PATH = os.path.join(ROOT, "runs")
MODELS_FOLDER = os.path.join(ROOT, "models")
MODELS_LIST = ("oaix_medicine_v1.pt", "yolo11m.pt")
MODEL_PATH = os.path.join(ROOT, MODELS_FOLDER, MODELS_LIST[0])
LOTE_DB = os.path.join(ROOT, 'db', 'og_lotes', 'Listado de lotes.xlsb')
RTSP_URL = os.getenv("RTSP_URL", "rtsp://172.23.23.15:8554/mystream_5")
print(f"{RTSP_URL=}")
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
    Ultra-low-buffer RTSP reader using OpenCV CAP_FFMPEG.
    - Forces FFmpeg low-latency demux/decoder options (no large queues).
    - Grabs continuously, drops everything except the most recent frame.
    - Avoids decode on every packet by using grab() to flush, retrieve() to publish.
    """
    def __init__(self, url: str, transport: str = "tcp"):
        self.url = url
        self.transport = transport  # "tcp" or "udp"
        self._lock = threading.Lock()
        self._cap: Optional[cv2.VideoCapture] = None
        self._latest: Optional[Tuple[float, np.ndarray]] = None  # (ts_ms, frame_bgr)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rtsp-reader", daemon=True)

        # small sleep to yield CPU in the inner loop (ms)
        self.read_sleep_ms = 1

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
        # Must be set BEFORE creating VideoCapture
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;tcp"           # or udp, but match your streamer
            "|stimeout;5000000"             # 5s (µs) socket timeout
            "|fflags;nobuffer"
            "|flags;low_delay"
            "|max_delay;0"
            "|probesize;32768"
            "|analyzeduration;0"
            "|use_wallclock_as_timestamps;1"
            "|reorder_queue_size;0"         # ignored if not supported; safe to keep
        )

        # Force the FFmpeg backend:
        self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            return False

        # Minimise internal queue if supported
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Warmup/flush: grab a handful of packets to clear any backlog
        for _ in range(max(0, RTSP_WARMUP_READS)):
            self._cap.grab()

        return True

    def _run(self):
        MAX_DRAIN = 8  # small, bounded flush per cycle
        while not self._stop.is_set():
            try:
                if not self._open():
                    time.sleep(0.5)
                    continue

                while not self._stop.is_set():
                    cap = self._cap
                    if cap is None:
                        print('break_1')
                        break
                    # Drain any backlog quickly (no decode cost)
                    drained = 0
                    while drained < MAX_DRAIN: 
                        ok = cap.grab()
                        # When there's nothing left immediately, stop draining
                        # (grab() returns quickly if no packet ready)
                        if not ok: 
                            print('break_2 (grab failed)')
                            break
                        drained += 1
                        # Now retrieve the most recent frame (decode once)
                        ok, frame = cap.retrieve()
                        if not ok or frame is None:
                            # camera hiccup -> reopen
                            print('break_3')
                            break

                        ts_ms = time.time() * 1000.0
                        with self._lock:
                            # keep only the freshest decoded frame
                            self._latest = (ts_ms, frame)

                        if self.read_sleep_ms > 0:
                            time.sleep(self.read_sleep_ms / 1000.0)

            except Exception:
                pass
            finally:
                try:
                    if self._cap is not None:
                        self._cap.release()
                except Exception:
                    pass
                self._cap = None
                time.sleep(0.5)  # brief backoff before reconnect

    def get_latest(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest is None:
                return None
            _, img = self._latest
        return img.copy()
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# create runs folder
utils.create_run_folder_output(SAVE_RUN_PATH, "run")
# Mount runs folder to serve predictions
app.mount("/runs", StaticFiles(directory=SAVE_RUN_PATH), name="runs")
# start RTSP Reader and streamer
reader = RTSPReader(RTSP_URL)
reader.start()
# Start Yolo prediction
detector = YoloDetect(MODEL_PATH)
# Start OCR Reader
ocr = OCR()
# Start LoteDB 
lote_db = LoteDB(LOTE_DB)
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


@app.get("/stream/mjpg")
def stream_mjpeg(fps: int = 15, quality: int = 80):
    interval = 1.0 / max(1, min(fps, 60))

    def gen():
        boundary = b"--frame"
        while True:
            frame = reader.get_latest()  # freshest frame from your persistent reader
            if frame is None:
                time.sleep(0.03)
                continue
            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            if not ok:
                continue
            chunk = jpg.tobytes()
            yield (
                boundary + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(chunk)).encode() + b"\r\n\r\n" +
                chunk + b"\r\n"
            )
            time.sleep(interval)

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


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
    tot_start = time.perf_counter() 
    start = time.perf_counter()
    frame = reader.get_latest()
    if frame is None:
        raise HTTPException(status_code=503, detail="No fresh frame available (stream not ready or stale)")
    stop = time.perf_counter()
    perf = stop - start
    _GET_LATEST_IN_S = utils.fmt_seconds(perf)
    print(f"GET_LATEST:{_GET_LATEST_IN_S}")
    # 2) YOLO detection
    start = time.perf_counter()
    boxes, confs, clids = detector.predict(frame)
    stop = time.perf_counter()
    perf = stop - start
    _PREDICT_IN_S = utils.fmt_seconds(perf)
    print(f"PREDICT:{_PREDICT_IN_S}")
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
    run_folder = utils.create_run_folder_output(SAVE_RUN_PATH, "run") if save_artifacts else None
    crop_path = full_path = final_path = None
    full_path = os.path.join(run_folder, utils.file_name("med_full"))
    cv2.imwrite(full_path, frame)
    crop_path = os.path.join(run_folder, utils.file_name("med_roi"))
    cv2.imwrite(crop_path, roi)
    # 3) OCR Detection
    start = time.perf_counter()
    optimized = utils.optimize_image_for_ocr(crop_path)
    result = ocr.gem_detect(optimized)
    stop = time.perf_counter()
    perf = stop - start
    _OAIXGOCR_IN_S = utils.fmt_seconds(perf)
    print(f"OAIXGOCR:{_OAIXGOCR_IN_S}")

    lot = result.get('lot_number')
    exp = result.get('expiry_date')

    if save_artifacts:
        final = roi.copy()
        cv2.putText(final, f"lot:{lot}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(final, f"exp:{exp}", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        final_path = os.path.join(run_folder, utils.file_name("final"))
        cv2.imwrite(final_path, final)

    # 4) Lote Match
    start = time.perf_counter()
    print(f'Lote search={lot}')
    lote_found = lote_db.find_lote(lote=lot)
    print(f'{lote_found=}')
    if lote_found is not None and not lote_found.empty:
        lote_match = 'Match'
    else:
        lote_match = 'No Match'
    stop = time.perf_counter()
    perf = stop - start
    _LOTE_SEARCH_IN_S = utils.fmt_seconds(perf)
    print(f"LOTE_SEARCH:{_LOTE_SEARCH_IN_S}")
    tot_stop = time.perf_counter()        
    tot_perf = tot_stop - tot_start
    _TOTAL_IN_S = utils.fmt_seconds(tot_perf)
    
    # 5) Completed, create return payload
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
            "lote_match": lote_match
        },
        "oaix_perf":{
            "get_latest_frame":_GET_LATEST_IN_S,
            "box_detection":_PREDICT_IN_S,
            "lote_expiry_predicition":_OAIXGOCR_IN_S,
            "lote_search":_LOTE_SEARCH_IN_S,
            "total_process_time":_TOTAL_IN_S
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
