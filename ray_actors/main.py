import ray
import os
import cv2
import time
import asyncio
from ultralytics import YOLO
try:
    from ray_actors.ocr_processor import OCRProcessor
except Exception:
    from ocr_processor import OCRProcessor

# Reset any stale Ray state and resource overrides, then initialize with whole-number resources
ray.shutdown()
os.environ.pop("RAY_OVERRIDE_RESOURCES", None)
os.environ.pop("RAY_num_gpus", None)
ray.init(ignore_reinit_error=True, num_cpus=4, num_gpus=1)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
@ray.remote
class DetectorActor():
    def __init__(self, rtsp_url:str, channel_name:str, model_path:str='yolo11m.pt'):
        self.rtsp_url = rtsp_url
        self.channel_name = channel_name
        self.channel_run = f"run-{int(time.time())}"
        self.using_fallback = False

        print(f"Initializing DetectorActor with RTSP URL: {rtsp_url}")
        # Reduce noisy OpenCV logs
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except Exception:
            pass
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            print(f"ERROR: Failed to open RTSP stream: {rtsp_url}. Falling back to sample video...")
            fallback_path = os.path.abspath(os.path.join(ROOT, 'rtsp_streamer', 'videos', 'emi_test.mp4'))
            self.cap = cv2.VideoCapture(fallback_path)
            if self.cap.isOpened():
                print(f"Using fallback video: {fallback_path}")
                self.using_fallback = True
            else:
                print(f"ERROR: Fallback video not available: {fallback_path}")
        else:
            print(f"Successfully opened RTSP stream: {rtsp_url}")
        self.capture_ready = self.cap.isOpened()
        
        # Initialize YOLO model (start on CPU for stability)
        self.model = YOLO(model_path)
        print(f"Loaded YOLO model: {model_path}")
        if hasattr(self.model, 'model'):
            try:
                actual_device = next(self.model.model.parameters()).device
                print(f"YOLO model device: {actual_device}")
            except Exception:
                print("Could not determine YOLO model device")
        
        # Initialize OCR processor
        print(f"Initializing OCR processor for {channel_name}...")
        # Force OCR to CPU to avoid CUDA abort in Ray workers
        self.ocr_processor = OCRProcessor(languages=['en'], gpu=False)
        print(f"OCR processor initialized on device: {self.ocr_processor.device}")

    def create_annotated_frame(self, frame, dets, ocr_results, save_rotated=False):
        """
        Create annotated frame with both YOLO detections and OCR results
        
        Args:
            frame: Original frame
            dets: YOLO detections
            ocr_results: OCR results
            save_rotated: Whether to save the frame rotated (for OCR display)
        """
        # Start with original frame or rotated frame
        if save_rotated:
            # Rotate frame 90 degrees clockwise for OCR display
            annotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        else:
            annotated_frame = frame.copy()
        
        # Draw YOLO bounding boxes (only on non-rotated frame)
        if not save_rotated:
            for det in dets:
                x1, y1, x2, y2 = det[0]
                cls_id = det[1]
                conf = det[2]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = self.model.names[cls_id]
                cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 2)
        
        # Draw OCR results using the OCR processor
        annotated_frame = self.ocr_processor.draw_ocr_results(annotated_frame, ocr_results)
        
        # Add processing info
        yolo_count = len(dets)
        ocr_count = ocr_results.get('text_count', 0)
        info_text = f"YOLO: {yolo_count} objects | OCR: {ocr_count} texts"
        cv2.putText(annotated_frame, info_text, (10, annotated_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame

    def save_frame(self, frame, dets, ocr_results):
        global ROOT
        import json
        
        timestamp = int(time.time())
        
        # Create directories if they don't exist
        save_dir = os.path.join(ROOT, self.channel_name, self.channel_run)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save normal frame with YOLO detections
        normal_filename = f"{self.channel_name}_normal_{timestamp}.jpg"
        normal_save_path = os.path.join(save_dir, normal_filename)
        normal_frame = self.create_annotated_frame(frame, dets, ocr_results, save_rotated=False)
        cv2.imwrite(normal_save_path, normal_frame)
        print(f"Normal frame saved: {normal_save_path}")
        
        # Save JSON file for normal frame
        normal_json_filename = f"{self.channel_name}_normal_{timestamp}.json"
        normal_json_path = os.path.join(save_dir, normal_json_filename)
        normal_data = {
            "timestamp": timestamp,
            "channel": self.channel_name,
            "yolo_detections": len(dets),
            "ocr_results": {
                "text_count": ocr_results.get('text_count', 0),
                "lot": ocr_results.get('lot', ''),
                "expiry": ocr_results.get('expiry', ''),
                "full_text": ocr_results.get('text', '')
            }
        }
        with open(normal_json_path, 'w') as f:
            json.dump(normal_data, f, indent=2)
        print(f"Normal JSON saved: {normal_json_path}")
        
        # Save rotated frame with OCR results (only if OCR found text)
        if ocr_results.get('text_count', 0) > 0:
            ocr_filename = f"{self.channel_name}_ocr_{timestamp}.jpg"
            ocr_save_path = os.path.join(save_dir, ocr_filename)
            ocr_frame = self.create_annotated_frame(frame, dets, ocr_results, save_rotated=True)
            cv2.imwrite(ocr_save_path, ocr_frame)
            print(f"OCR frame saved (rotated): {ocr_save_path}")
            
            # Save JSON file for OCR frame
            ocr_json_filename = f"{self.channel_name}_ocr_{timestamp}.json"
            ocr_json_path = os.path.join(save_dir, ocr_json_filename)
            ocr_data = {
                "timestamp": timestamp,
                "channel": self.channel_name,
                "frame_type": "ocr_rotated",
                "yolo_detections": len(dets),
                "ocr_results": {
                    "text_count": ocr_results.get('text_count', 0),
                    "lot": ocr_results.get('lot', ''),
                    "expiry": ocr_results.get('expiry', ''),
                    "full_text": ocr_results.get('text', ''),
                    "lines": ocr_results.get('lines', [])
                }
            }
            with open(ocr_json_path, 'w') as f:
                json.dump(ocr_data, f, indent=2)
            print(f"OCR JSON saved: {ocr_json_path}")
        
        print(f"Saved {len(dets)} YOLO detections and {ocr_results.get('text_count', 0)} OCR texts")
        return normal_filename

    def step(self):
        if not self.capture_ready:
            return {"event":"error", "msg":"capture_not_opened"}
        ok, frame = self.cap.read()
        if not ok and self.using_fallback:
            # Loop fallback video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok:
            return {"event":"error", "msg":"read_failed"}
        
        # YOLO Object Detection
        start_time = time.time()
        res = self.model(frame, imgsz=640, conf=0.25, verbose=False, device='cpu')
        yolo_time = time.time() - start_time
        
        # OCR Text Recognition (with iPhone rotation)
        ocr_start_time = time.time()
        ocr_results = self.ocr_processor.process_frame(frame, rotate_iphone=True)
        ocr_time = time.time() - ocr_start_time
        
        # Basic timings (no CUDA calls)
        print(f"YOLO inference time: {yolo_time*1000:.1f} ms | OCR: {ocr_time*1000:.1f} ms | OCR texts: {ocr_results.get('text_count', 0)}")
        
        # Process YOLO detections
        dets = []
        for r in res:
            if r.boxes is None: continue
            for b in r.boxes:
                 dets.append((
                       [int(round(x)) for x in b.xyxy[0].tolist()],
                       int(b.cls.item()),
                       float(b.conf.item())
                   ))
        
        # Save frames with annotations (both normal and rotated OCR versions)
        self.save_frame(frame, dets, ocr_results)
        
        return {
            "event": "detections", 
            "data": {
                "yolo_detections": dets,
                "ocr_results": ocr_results,
                "processing_times": {
                    "yolo_ms": yolo_time * 1000,
                    "ocr_ms": ocr_time * 1000,
                    "total_ms": (yolo_time + ocr_time) * 1000
                }
            }
        }


    def close(self):
        self.cap.release()
        return True


# create actors
urls = [
    'rtsp://172.23.23.15:8554/mystream_4'
    # 'rtsp://172.23.23.15:8554/mystream_2',
    # 'rtsp://172.23.23.15:8554/mystream_3'
]

actors = []
for i, url in enumerate(urls):
    actors.append(DetectorActor.remote(url, f"channel_{i+1}"))

async def poll():
    print("Starting detection loop...")
    while True:
        futures = [a.step.remote() for a in actors]
        # Wait for all Ray futures to complete
        results = ray.get(futures)
        
        # Print results for debugging
        for i, result in enumerate(results):
            if result.get("event") == "detections":
                data = result.get("data", {})
                yolo_detections = data.get("yolo_detections", [])
                ocr_results = data.get("ocr_results", {})
                processing_times = data.get("processing_times", {})
                
                if yolo_detections or ocr_results.get("text_count", 0) > 0:
                    print(f"Actor {i}: YOLO={len(yolo_detections)} objects, OCR={ocr_results.get('text_count', 0)} texts")
                    print(f"  Processing times: YOLO={processing_times.get('yolo_ms', 0):.1f}ms, OCR={processing_times.get('ocr_ms', 0):.1f}ms")
                    
                    # Show first few YOLO detections
                    for det in yolo_detections[:2]:
                        bbox, cls_id, conf = det
                        print(f"    - Class {cls_id}: {conf:.2f} confidence")
                    
                    # Show OCR text if found
                    if ocr_results.get("text"):
                        text_preview = ocr_results["text"][:100] + "..." if len(ocr_results["text"]) > 100 else ocr_results["text"]
                        print(f"    - OCR text: {text_preview}")
                        
            elif result.get("event") == "error":
                print(f"Actor {i}: Error - {result.get('msg')}")
        
        # forward results to your gRPC clients / WS
        # (or store in Redis/Kafka)
        await asyncio.sleep(0.01)

asyncio.run(poll())

terminate = input("terminate>")
print(f"exiting . .")