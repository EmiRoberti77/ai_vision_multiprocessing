import ray
import os
import cv2
import time
import asyncio
from ultralytics import YOLO

ray.init(num_cpus=4, num_gpus=1)


@ray.remote(num_gpus=1)
class DetectorActor():
    def __init__(self, rtsp_url:str, model_path:str='yolo11m.pt'):
        self.rtsp_url = rtsp_url
        print(f"Initializing DetectorActor with RTSP URL: {rtsp_url}")
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            print(f"ERROR: Failed to open RTSP stream: {rtsp_url}")
        else:
            print(f"Successfully opened RTSP stream: {rtsp_url}")
        self.model = YOLO(model_path)
        print(f"Loaded YOLO model: {model_path}")

    def step(self):
        ok, frame = self.cap.read()
        if not ok:
            return {"event":"error", "msg":"read_failed"}
        
        print(f"Processing frame from {self.rtsp_url}")
        res = self.model(frame, imgsz=640, conf=0.25, verbose=False)
        dets = []
        for r in res:
            if r.boxes is None: continue
            for b in r.boxes:
                 dets.append((
                    [float(x) for x in b.xyxy[0].tolist()],
                    int(b.cls.item()),
                    float(b.conf.item())
                ))

        return {"event":"detections","data":dets}

    def close(self):
        self.cap.release()
        return True


# create actors
url = ['rtsp://172.23.23.15:8554/mystream']
actors = [DetectorActor.remote(url[0])]

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
                    print(f"Actor {i}: Found {len(detections)} detections")
                    for det in detections[:3]:  # Show first 3 detections
                        bbox, cls_id, conf = det
                        print(f"  - Class {cls_id}: {conf:.2f} confidence at {bbox}")
                else:
                    print(f"Actor {i}: No detections")
            elif result.get("event") == "error":
                print(f"Actor {i}: Error - {result.get('msg')}")
        
        # forward results to your gRPC clients / WS
        # (or store in Redis/Kafka)
        await asyncio.sleep(0.01)

asyncio.run(poll())

terminate = input("terminate>")
print(f"exiting . .")