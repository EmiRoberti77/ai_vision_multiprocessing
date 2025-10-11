import json
import utils
import os
from ultralytics import YOLO
import cv2
from OAIX_GOCR_Detection import OAIX_GOCR_Detection as OCR
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
SAVE_RUN_PATH = os.path.join(ROOT, 'runs')
MODELS_LIST = ('oaix_medicine_v1.pt', 'yolo11m.pt')
IMAGES_LIST = os.listdir(os.path.join(ROOT, 'images'))
print(IMAGES_LIST)
MODELS_FOLDER = 'models'
IMAGES_FOLEDR = os.path.join(ROOT, 'images') 
MODEL_PATH = os.path.join(ROOT, MODELS_FOLDER, MODELS_LIST[0])

class YoloDetect():
    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)

    def class_name(self, clsid:int)->str:
        return self.model.names[clsid]

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

    def crop_frame_roi(self, frame, x1, x2, y1, y2):
        crop_frame = []
        if x2 > x1 and y2 > y1:
           crop_frame = frame[y1:y2, x1:x2].copy()
        
        return crop_frame


def main():
    detect = YoloDetect(model_path=MODEL_PATH)
    run_folder = utils.create_run_folder_output(SAVE_RUN_PATH, 'run')
    for image in IMAGES_LIST:
        frame = cv2.imread(os.path.join(IMAGES_FOLEDR, image))
        boxes, confs, clids = detect.predict(frame)
        for box, conf, clid in zip(boxes, confs, clids):
            x1, y1, x2, y2 = box
            print(x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, f"{conf}:{clid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
            crop_frame = detect.crop_frame_roi(frame=frame, x1=x1, x2=x2, y1=y1, y2=y2)
            
            crop_frame_path = os.path.join(run_folder, utils.file_name('med_roi'))
            cv2.imwrite(crop_frame_path, crop_frame)

            full_frame_path = os.path.join(run_folder, utils.file_name('med_full'))
            cv2.imwrite(full_frame_path, frame)
            
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
            
            print(result)
            
    return 0


if __name__ == "__main__":
    main()