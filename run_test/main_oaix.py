import json
import utils
import os
from ultralytics import YOLO
import cv2
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
SAVE_RUN_PATH = os.path.join(ROOT, 'runs')
MODELS_LIST = ('oaix_medicine_v1.pt', 'yolo11m.pt')
IMAGES_LIST = ('box_1.png', 'box_2.png', 'box_3.png', 'box_4.png')
MODELS_FOLDER = 'models'
IMAGES_FOLEDR = os.path.join(ROOT, 'images') 
MODEL_PATH = os.path.join(ROOT, MODELS_FOLDER, MODELS_LIST[0])

class GeminiDetection():
    def __init__(self) -> None:
        if not os.getenv('OAIX_G_OCR'):
            raise ValueError("Error:Missing OAIX_G_OCR")

        # Configure the Gemini API
        genai.configure(api_key=os.getenv('OAIX_G_OCR'))
        
        # Create the model with system instruction
        self.model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            generation_config={
                "response_mime_type": "application/json"
            },
            system_instruction=(
                "You are an expert data extractor. Your task is to analyze the provided "
                "medicine label image and extract the Lot/Batch number and the Expiry Date. "
                "You must return the result as a single JSON object. "
                "Use 'lot_number' and 'expiry_date' as the keys. "
                "Format the expiry_date in YYYY-MM-DD format if possible."
            )
        )
    
    def gem_detect(self, image_path):
        img = Image.open(image_path)
        
        prompt = "Extract the Lot/Batch number and Expiry Date from this medicine package. Return as JSON with keys 'lot_number' and 'expiry_date'."
        
        response = self.model.generate_content([prompt, img])
        
        try:
            extracted_data = json.loads(response.text)
            return extracted_data
        except json.JSONDecodeError:
            print("Error: Gemini did not return valid JSON.")
            print("Raw response:", response.text)
            return None


class YoloDetect():
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

    def crop_frame_roi(self, frame, x1, x2, y1, y2):
        crop_frame = []
        if x2 > x1 and y2 > y1:
           crop_frame = frame[y1:y2, x1:x2].copy()
        
        return crop_frame


def main():
    detect = YoloDetect(model_path=MODEL_PATH)
    for image in IMAGES_LIST:
        frame = cv2.imread(os.path.join(IMAGES_FOLEDR, image))
        boxes, confs, clids = detect.predict(frame)
        for box, conf, clid in zip(boxes, confs, clids):
            x1, y1, x2, y2 = box
            print(x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, f"{conf}:{clid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        crop_frame = detect.crop_frame_roi(frame=frame, x1=x1, x2=x2, y1=y1, y2=y2)
        crop_frame_path = os.path.join(utils.create_run_folder_output(SAVE_RUN_PATH, 'run'), utils.file_name('med_roi'))
        print(f"{crop_frame_path=}")
        cv2.imwrite(crop_frame_path, crop_frame)
        cv2.imwrite(os.path.join(utils.create_run_folder_output(SAVE_RUN_PATH, 'run'), utils.file_name('med_full')), frame)
        
        gemini = GeminiDetection()
        result = gemini.gem_detect(crop_frame_path)
        print(result)
            
    return 0


if __name__ == "__main__":
    main()