import ray
import os
import cv2
import time
import asyncio
from ultralytics import YOLO

ray.init(num_cpus=4, num_gpus=1)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
@ray.remote(num_gpus=0.25)
class DetectorActor():
    def __init__(self, rtsp_url:str, channel_name:str, model_path:str='yolo11m.pt'):
        self.rtsp_url = rtsp_url
        self.channel_name = channel_name
        self.channel_run = f"run-{int(time.time())}"

        print(f"Initializing DetectorActor with RTSP URL: {rtsp_url}")
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            print(f"ERROR: Failed to open RTSP stream: {rtsp_url}")
        else:
            print(f"Successfully opened RTSP stream: {rtsp_url}")
        
        # Check CUDA availability
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name()}")
        
        self.model = YOLO(model_path)
        print(f"Loaded YOLO model: {model_path}")
        
        # Check if model is on GPU
        if hasattr(self.model, 'model'):
            device = next(self.model.model.parameters()).device
            print(f"YOLO model device: {device}")
        else:
            print("Could not determine YOLO model device")

    def save_frame(self, frame, dets):
        global ROOT
        # Save frame with optional overlay
        timestamp = int(time.time())
        filename = f"{self.channel_name}_{timestamp}.jpg"
        save_path = os.path.join(ROOT, self.channel_name, self.channel_run, filename)
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"Attempting to save to: {save_path}")
        
        # draw bounding boxes
        for det in dets:
            x1, y1, x2, y2 = det[0]
            cls_id = det[1]
            conf = det[2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255,0) , 2)
            label = self.model.names[cls_id]
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 0.9, (0,255,0), 2)
        
        cv2.imwrite(save_path, frame)
        print(f"frame saved {save_path}")
        return filename

    def step(self):
        ok, frame = self.cap.read()
        if not ok:
            return {"event":"error", "msg":"read_failed"}
        
        # Monitor GPU before inference
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU cache
            gpu_memory_before = torch.cuda.memory_allocated() / 1024**2  # MB
            print(f"GPU memory before inference: {gpu_memory_before:.1f} MB")
        
        # Time the inference
        start_time = time.time()
        res = self.model(frame, imgsz=640, conf=0.25, verbose=False)
        inference_time = time.time() - start_time
        
        # Monitor GPU after inference
        if torch.cuda.is_available():
            gpu_memory_after = torch.cuda.memory_allocated() / 1024**2  # MB
            print(f"GPU memory after inference: {gpu_memory_after:.1f} MB")
            print(f"Inference time: {inference_time*1000:.1f} ms")
        
        dets = []
        for r in res:
            if r.boxes is None: continue
            for b in r.boxes:
                 dets.append((
                       [int(round(x)) for x in b.xyxy[0].tolist()],
                       int(b.cls.item()),
                       float(b.conf.item())
                   ))
                 
        self.save_frame(frame, dets)
        return {"event":"detections","data":dets}


    def close(self):
        self.cap.release()
        return True


# create actors
urls = [
    'rtsp://172.23.23.15:8554/mystream_1',
    'rtsp://172.23.23.15:8554/mystream_2',
    'rtsp://172.23.23.15:8554/mystream_3'
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
                detections = result.get("data", [])
                if detections:
                    #print(f"Actor {i}: Found {len(detections)} detections")
                    for det in detections[:3]:  # Show first 3 detections
                        bbox, cls_id, conf = det
                        #print(f"  - Class {cls_id}: {conf:.2f} confidence at {bbox}")
            elif result.get("event") == "error":
                print(f"Actor {i}: Error - {result.get('msg')}")
        
        # forward results to your gRPC clients / WS
        # (or store in Redis/Kafka)
        await asyncio.sleep(0.01)

asyncio.run(poll())

terminate = input("terminate>")
print(f"exiting . .")