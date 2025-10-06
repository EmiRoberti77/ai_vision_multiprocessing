
import os
import cv2
import json
import numpy as np
from med_pack_extractor import MedPackExtractor
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
SAVE_RUN_PATH = os.path.join(ROOT, 'runs')
MODELS_LIST = ('oaix_medicine_v1.pt', 'yolo11m.pt')
IMAGES_LIST = ('box_1.png', 'box_2.png', 'box_3.png', 'box_4.png')
MODELS_FOLDER = 'models'
IMAGES_FOLEDR = os.path.join(ROOT, 'images') 
MODEL_PATH = os.path.join(ROOT, MODELS_FOLDER, MODELS_LIST[0])

def load_bgr(p):
    img = cv2.imread(p)
    if img is None:
        raise RuntimeError(f"Failed to read {p}")
    return img

def main():
    extractor = MedPackExtractor(use_paddle=None, paddle_lang='en', tesseract_lang='eng+fra+deu+spa+ita+por')
    samples = [
        os.path.join(ROOT, 'images', 'box_1.png'),
        os.path.join(ROOT, 'images', 'box_2.png'),
        os.path.join(ROOT, 'images', 'box_3.png'),
        os.path.join(ROOT, 'images', 'box_4.png'),
    ]
    results = {}
    for p in samples:
        if not os.path.exists(p):
            continue
        frame = load_bgr(p)
        pred = extractor.extract(frame)
        results[os.path.basename(p)] = pred

    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # If OCR backends aren't installed in this environment, note it gracefully.
        print(f"Demo could not run: {e}")
