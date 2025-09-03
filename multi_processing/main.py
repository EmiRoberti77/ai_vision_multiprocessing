"""
Multi-processing supervisor ( no external infra )
Use this pattern when on a single machine ( 1 - 2 GPU or CPU )
with 5 to 30 streams.  attach to a gRPC server to send commands
"""

from tabnanny import verbose
import os, signal, time
from multiprocessing import Process, Queue, Manager
from queue import Empty
from typing import Dict
import torch, cv2
from torch.cpu import stream
from ultralytics import YOLO


def run_supervisor(num_workers=2, gpus=(0,)):
    manager = Manager()
    cmd_qs, result_q = [], manager.Queue(maxsize=1000)
    workers = []
    # start N workers; map to GPUs round robin
    for i in range(num_workers):
        q = manager.Queue(maxsize=200)
        cmd_qs.append(q)
        p = Process(target=detector_worker, args=(q, result_q, gpus[i % len(gpus)]), daemon=True)
        workers.append(p)
        p.start()  # Start the worker process

    # example: assign streams to workers by simple hash
    def dispatch(cmd):
        sid = cmd.get("stream_id", "")
        idx = hash(sid) % num_workers
        cmd_qs[idx].put(cmd)

    return dispatch, result_q, workers, cmd_qs


def detector_worker(cmd_q:Queue, result_q:Queue, gpu_id:int=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    model = YOLO('yolo11m.pt')
    streams = {}
    
    while True:
        try:
            cmd = cmd_q.get(timeout=0.1)  # Use timeout instead of get_nowait()
            
            if cmd["type"] == "START":
                sid, url = cmd["stream_id"], cmd["rtsp"]  # Fixed typo: rstp -> rtsp
                cap = cv2.VideoCapture(url)
                streams[sid] = cap
                print(f"Started stream {sid}")
                
            elif cmd["type"] == "STOP":
                sid = cmd["stream_id"]
                if sid in streams:
                    streams[sid].release()
                    del streams[sid]
                    print(f"Stopped stream {sid}")
                    
            elif cmd["type"] == "SHUTDOWN":
                break
                
        except Empty:
            pass  # No commands, continue to process streams
        
        # Process all active streams
        for sid, cap in list(streams.items()):
            ok, frame = cap.read()
            if not ok:
                result_q.put({"stream_id":sid, "event":"error", "msg":"read_failed"})
                cap.release()
                del streams[sid]
                continue
            
            results = model(frame, imgsz=640, conf=0.25, verbose=False)
            dets = []
            for r in results:
                if r is None: continue
                for b in r.boxes:
                    xyxy = b.xyxy[0].tolist()
                    cls = int(b.cls.item())
                    conf = float(b.conf.item())
                    dets.append((xyxy, cls, conf))
            
            if dets:  # Only send if there are detections
                result_q.put({"stream_id": sid, "event": "detections", "data": dets})
        
        time.sleep(0.001)  # tiny yield

if __name__ == "__main__":
    print("Starting")
    dispatch, result_q, workers, _ = run_supervisor(num_workers=2, gpus=(0,))
    print("run_supervisor complete")
    # example commands
    dispatch({"type":"START","stream_id":"cam1","rtsp":"rtsp://172.23.23.15:8554/mystream"})
    dispatch({"type":"START","stream_id":"cam2","rtsp":"rtsp://172.23.23.15:8554/mystream"})
    print("commands dispatched")

    try:
        while True:
            print("in while loop")
            msg = result_q.get()
            # forward to gRPC subscribers / websockets / DB
            print(msg)
    except KeyboardInterrupt:
        pass