import numpy as np, cv2
from random import randint
import re
import easyocr
from datetime import datetime 
from ocr.common import clean_line, collapse_spaced_digits, find_lot_on_line, parse_expiry_from_text, has_exp_key

try:
    import torch
    use_gpu = torch.cuda.is_available()
    print("use_gpu", use_gpu)
except Exception:
    use_gpu = False

reader = easyocr.Reader(['en'], gpu=use_gpu)
print("Torch CUDA available:", torch.cuda.is_available())
print("EasyOCR reader.device:", getattr(reader, "device", "n/a"))

# Check where the models actually live
det_dev = next(reader.detector.parameters()).device
rec_dev = next(reader.recognizer.parameters()).device
print("Detector device:", det_dev)
print("Recognizer device:", rec_dev)

def _preprocess(bgr: np.ndarray) -> np.ndarray:
    if bgr is None or bgr.size == 0:
        return bgr
    h, w = bgr.shape[:2]
    if max(h, w) < 320:
        bgr = cv2.resize(bgr, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0, (8,8)).apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def ocr_and_parse(bgr: np.ndarray):
    proc = _preprocess(bgr)
    rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
    results = reader.readtext(rgb, detail=1, paragraph=False)

    # lines: keep bbox so you can debug/overlay later if you want
    # results item: [bbox, text, conf]
    raw_lines = [(t, float(c or 0.0)) for _, t, c in results]
    # clean + filter
    lines = [(clean_line(t), c) for t, c in raw_lines if (c or 0.0) >= 0.35 and t.strip()]

    # Join for debugging
    joined = " ".join([t for t, _ in lines])

    lot = ""
    expiry = ""

    # 1) Try to find LOT on any line with a lot key
    for t, _ in lines:
        lot_candidate = find_lot_on_line(t)
        if lot_candidate:
            # simple sanity: drop separators inside
            lot = re.sub(r"[^A-Z0-9\-_]", "", lot_candidate)
            break

    # 2) Try to find expiry on lines that look like expiry
    for t, _ in lines:
        if has_exp_key(t):
            expiry = parse_expiry_from_text(t)
            if expiry:
                break

    # 3) Fallback: look globally for a date if not found on keyed line
    if not expiry:
        up = clean_line(joined)
        expiry = parse_expiry_from_text(up)

    result = {
        "text": joined,
        "lot": lot,
        "expiry": expiry,
        "lines": lines,  # cleaned lines with conf
    }
    # print(result)
    return result