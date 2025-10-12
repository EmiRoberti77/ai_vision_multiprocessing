import json
import utils
import os
from ultralytics import YOLO
import cv2
from OAIX_GOCR_Detection import OAIX_GOCR_Detection as OCR
from PIL import Image
from dotenv import load_dotenv
import hashlib
import numpy as np
load_dotenv()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
SAVE_RUN_PATH = os.path.join(ROOT, 'runs')
MODELS_LIST = ('oaix_medicine_v1.pt', 'yolo11m.pt')
INPUT_VIDEO = os.path.join(ROOT, '..', 'rtsp_streamer', 'videos', 'Medicinas_rotated_180_1.mp4')
print(INPUT_VIDEO)
MODELS_FOLDER = 'models'
IMAGES_FOLEDR = os.path.join(ROOT, 'images') 
MODEL_PATH = os.path.join(ROOT, MODELS_FOLDER, MODELS_LIST[0])

class BoxTracker:
    def __init__(self, iou_threshold=0.5, min_confidence=0.6, stability_frames=5):
        self.tracked_boxes = {}  # {box_id: {'box': [x1,y1,x2,y2], 'confidence': float, 'frame_count': int, 'processed': bool, 'best_crop': None}}
        self.iou_threshold = iou_threshold
        self.min_confidence = min_confidence
        self.stability_frames = stability_frames
        self.next_id = 0
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union of two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_sharpness(self, crop):
        """Calculate image sharpness using Laplacian variance"""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def update_tracks(self, boxes, confs, clids, frame):
        """Update tracked boxes with new detections"""
        current_detections = list(zip(boxes, confs, clids))
        
        # Match detections to existing tracks
        matched_tracks = set()
        
        for box, conf, clid in current_detections:
            if conf < self.min_confidence:
                continue
                
            best_match_id = None
            best_iou = 0
            
            # Find best matching existing track
            for track_id, track_data in self.tracked_boxes.items():
                if track_data['processed']:
                    continue
                    
                iou = self.calculate_iou(box, track_data['box'])
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match_id = track_id
            
            if best_match_id is not None:
                # Update existing track
                track = self.tracked_boxes[best_match_id]
                track['frame_count'] += 1
                
                # Update with higher confidence detection
                if conf > track['confidence']:
                    track['box'] = box
                    track['confidence'] = conf
                    track['clid'] = clid
                    
                    # Update best crop if this one is sharper
                    x1, y1, x2, y2 = box
                    crop = frame[y1:y2, x1:x2].copy()
                    if crop.size > 0:
                        sharpness = self.calculate_sharpness(crop)
                        if track['best_crop'] is None or sharpness > track['best_sharpness']:
                            track['best_crop'] = crop
                            track['best_sharpness'] = sharpness
                
                matched_tracks.add(best_match_id)
            else:
                # Create new track
                x1, y1, x2, y2 = box
                crop = frame[y1:y2, x1:x2].copy()
                sharpness = self.calculate_sharpness(crop) if crop.size > 0 else 0
                
                self.tracked_boxes[self.next_id] = {
                    'box': box,
                    'confidence': conf,
                    'clid': clid,
                    'frame_count': 1,
                    'processed': False,
                    'best_crop': crop if crop.size > 0 else None,
                    'best_sharpness': sharpness
                }
                self.next_id += 1
    
    def get_ready_for_processing(self):
        """Get boxes that are stable and ready for OCR processing"""
        ready_boxes = []
        
        for track_id, track_data in self.tracked_boxes.items():
            if (not track_data['processed'] and 
                track_data['frame_count'] >= self.stability_frames and
                track_data['best_crop'] is not None):
                
                ready_boxes.append((track_id, track_data))
                track_data['processed'] = True  # Mark as processed
        
        return ready_boxes

class YoloDetect():
    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)

    def class_name(self, clsid:int)->str:
        return self.model.names[clsid]
    
    def crop_frame_roi(self, frame, x1, x2, y1, y2):
        crop_frame = []
        if x2 > x1 and y2 > y1:
           crop_frame = frame[y1:y2, x1:x2].copy()
        
        return crop_frame

    def predict(self, frame):
        boxes = []
        confs = []
        clids = [] 
        results = self.model.predict(frame, iou=0.4, conf=0.3, verbose=False)
        for r in results:
            for b in r.boxes:
                boxes.append(list(map(int, b.xyxy[0].tolist())))
                confs.append(round(float(b.conf), 2))
                clids.append(int(b.cls))
        
        return boxes, confs, clids

def main():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    detect = YoloDetect(model_path=MODEL_PATH)
    tracker = BoxTracker(iou_threshold=0.5, min_confidence=0.6, stability_frames=10)
    run_folder = utils.create_run_folder_output(SAVE_RUN_PATH, 'run')
    
    frame_count = 0
    processed_boxes = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{fps=}")
    frame_interval = int(fps) if fps > 0 else 25

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ok, frame = cap.read()
        if not ok:
            break
        
        frame_count += 1
        boxes, confs, clids = detect.predict(frame)
        
        for box, conf, clid in zip(boxes, confs, clids):
            processed_boxes += 1
            x1, y1, x2, y2 = box
            print(x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, f"{conf}:{clid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            crop_frame = detect.crop_frame_roi(frame=frame, x1=x1, x2=x2, y1=y1, y2=y2)
            # Save the crop
            crop_frame_path = os.path.join(run_folder, utils.file_name('med_roi'))
            cv2.imwrite(crop_frame_path, crop_frame)

            full_frame_path = os.path.join(run_folder, utils.file_name('med_full'))
            cv2.imwrite(full_frame_path, frame)
            # Run OCR on the high-quality crop
            ocr = OCR()
            result = ocr.gem_detect(crop_frame_path)
            
            final_frame = cv2.putText(crop_frame, f"lot:{result['lot_number']}", (10,30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,0),1,cv2.LINE_AA)
            final_frame = cv2.putText(final_frame, f"exp:{result['expiry_date']}", (10,60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,0),1,cv2.LINE_AA)
            final_frame_path = os.path.join(run_folder, utils.file_name('final'))
            cv2.imwrite(final_frame_path, final_frame)
            if result["lot_number"] is None or result["expiry_date"] is None:
                print(f"{crop_frame_path=}")
                print(f"{full_frame_path=}")
                print(f"{final_frame_path=}")
            

        frame_count += frame_interval
        print(f"{frame_count=}")
        # Show progress every 100 frames
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames, found {processed_boxes} unique boxes")
    
    print(f"Final: Processed {frame_count} frames, found {processed_boxes} unique boxes")
    cap.release()
    return 0

if __name__ == "__main__":
    main()