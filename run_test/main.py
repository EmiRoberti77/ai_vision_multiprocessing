import os
from ultralytics import YOLO
import cv2
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
SAVE_RUN_PATH = os.path.join(ROOT, 'runs')
MODELS_LIST = ('oaix_medicine_v1.pt', 'yolo11m.pt')
IMAGES_LIST = ('box_1.png', 'box_2.png', 'box_3.png', 'box_4.png')
MODELS_FOLDER = 'models'
IMAGES_FOLEDR = os.path.join(ROOT, 'images') 
MODEL_PATH = os.path.join(ROOT, MODELS_FOLDER, MODELS_LIST[0])
import utils
print(f"{ROOT=}")
print(f"{MODEL_PATH}")

class Detect():
    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)

    def class_name(self, clsid:int)->str:
        return self.model.names[clsid]

    def predict(self, frame):
        boxes = []
        confs = []
        clids = [] 
        results = self.model.predict(frame, iou=0.4, conf=0.3)
        for r in results:
            for b in r.boxes:
                boxes.append(list(map(int, b.xyxy[0].tolist())))
                confs.append(round(float(b.conf), 2))
                clids.append(int(b.cls))
        
        return boxes, confs, clids



def main():
    detect = Detect(model_path=MODEL_PATH)
    frame = cv2.imread(os.path.join(IMAGES_FOLEDR, IMAGES_LIST[0]))
    boxes, confs, clids,  = detect.predict(frame)
    for box, conf, clid in zip(boxes, confs, clids):
        x1, y1, x2, y2 = box;
        print(x1, y1, x2, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(frame, f"{conf}:{clid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    save_run_path = os.path.join(SAVE_RUN_PATH, utils.folder_name('run'))
    if not os.path.exists(save_run_path):
        os.makedirs(save_run_path)

    cv2.imwrite(os.path.join(save_run_path, utils.file_name('med')), frame)
        
    return 0

if __name__ == "__main__":
    main()



