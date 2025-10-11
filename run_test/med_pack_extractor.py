
import re
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Optional deps (use if installed)
try:
    from paddleocr import PaddleOCR  # type: ignore
    _HAS_PADDLE = True
except Exception:
    _HAS_PADDLE = False
print(f"{_HAS_PADDLE=}")

try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
    _HAS_TESS = True
except Exception:
    _HAS_TESS = False
    
print(f"{_HAS_TESS=}")

import numpy as np

@dataclass
class OCRBox:
    text: str
    conf: float
    box: Tuple[int, int, int, int]  # x1,y1,x2,y2

@dataclass
class FieldPrediction:
    value: Optional[str]
    score: float
    box: Optional[Tuple[int, int, int, int]]
    source_text: Optional[str]

class MedPackExtractor:
    """
    Extracts LOT/BATCH and EXP/BBE dates from medicine packaging.
    - Input: a single image frame as a numpy array (H,W,3) BGR or RGB.
    - Output: dict with 'lot' and 'expiry' FieldPrediction + debug info
    """
    # Common multilingual prefixes/synonyms
    LOT_PREFIXES = [
        # English
        r"\bLOT\b", r"\bBATCH\b", r"\bBATCH\s*NO\.?\b", r"\bLOT\s*NO\.?\b",
        # French
        r"\bLOT\b", r"\bN[°o]\s*LOT\b", r"\bN[°o]\b",
        # Spanish/Portuguese
        r"\bLOTE\b", r"\bLOTE\s*NO\.?\b",
        # Italian
        r"\bLOTTO\b", r"\bN[°o]\s*LOTTO\b",
        # German / Dutch
        r"\bCHARGE\b", r"\bCHARGEN?-?NR\.?\b", r"\bBATCH\s*NR\.?\b", r"\bPARTIJ\b",
        # Polish / Romanian / Czech / Slovak (common variants)
        r"\bSERIA\b", r"\bPARTIA\b", r"\bLOT\s*NR\.?\b",
    ]

    EXP_PREFIXES = [
        # English
        r"\bEXP\b", r"\bEXP\.?\s*DATE\b", r"\bEXPI?R?Y?\b", r"\bUSE\s*BY\b", r"\bBEST\s*BEFORE\b",
        r"\bBBE\b",
        # French
        r"\bDLUO\b", r"\bDLC\b", r"\bDATE\s*DE\s*P[EÉ]REMPTION\b",
        # Spanish / Portuguese / Italian
        r"\bCAD\b", r"\bCADUCIDAD\b", r"\bVENC\.?\b", r"\bVENCE\b", r"\bSCAD(?:ENZA)?\b",
        # German / Dutch
        r"\bMHD\b", r"\bHALTBAR(?:KEIT)?\b", r"\bMINDESTHALTBARKEITS?DATUM\b", r"\bTENMINSTE\s*HOUDBAAR\s*TOT\b",
        # Nordics / others (common)
        r"\bUTG\.?\b", r"\bUTL[BÐ]P\.?\b", r"\bP[EĘ]RIM\.?\b"
    ]

    # Date patterns: accept many separators and orders. Capture year to disambiguate.
    DATE_PATTERNS = [
        # 31/12/2026, 31-12-26, 31.12.2026
        r"\b([0-3]?\d)[/\-\. ]([0-1]?\d)[/\-\. ]((?:20)?\d{2})\b",
        # 12/2026 or 12-26
        r"\b([0-1]?\d)[/\-\. ]((?:20)?\d{2})\b",
        # 2026-12 or 2026.12
        r"\b((?:20)?\d{2})[/\-\. ]([0-1]?\d)\b",
        # Mon YYYY (English)
        r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[\.\- ]+((?:20)?\d{2})\b",
        # MMM YYYY (many langs)
        r"\b(ENE|FEB|MAR|ABR|MAY|JUN|JUL|AGO|SEP|OCT|NOV|DIC)[\.\- ]+((?:20)?\d{2})\b",  # ES
        r"\b(GEN|FEB|MAR|APR|MAG|GIU|LUG|AGO|SET|OTT|NOV|DIC)[\.\- ]+((?:20)?\d{2})\b",  # IT
        r"\b(JAN|FEB|MÄR|MA[R|I]|APR|MAJ|JUN|JUL|AUG|SEP|OKT|NOV|DEZ)[\.\- ]+((?:20)?\d{2})\b",  # DE/NL variants
    ]

    # Looser code pattern often seen for LOT codes: alnum with possible slash/dash
    LOT_CODE_PATTERN = r"\b([A-Z0-9]{2,4}[-/ ]?[A-Z0-9]{3,8})\b"

    def __init__(self,
                 use_paddle: Optional[bool] = None,
                 paddle_lang: str = 'en',
                 tesseract_lang: str = 'eng',
                 det_db_box_thresh: float = 0.5):
        """
        use_paddle: force PaddleOCR (True/False) or auto-detect if None
        paddle_lang: PaddleOCR lang code (e.g., 'en', 'fr', 'de', 'latin')
        tesseract_lang: pytesseract languages (e.g., 'eng+fra+deu')
        """
        self._engine = None
        self._engine_name = "none"

        if use_paddle is None:
            use_paddle = _HAS_PADDLE or (not _HAS_TESS)

        if use_paddle and _HAS_PADDLE:
            # Initialize PaddleOCR detector+recognizer
            self._engine = PaddleOCR(lang=paddle_lang, det_db_box_thresh=det_db_box_thresh)
            self._engine_name = "paddleocr"
        elif _HAS_TESS:
            self._engine = "tesseract"  # sentinel
            self._tess_lang = tesseract_lang
            self._engine_name = "tesseract"
        else:
            raise RuntimeError("No OCR backend available. Install paddleocr or pytesseract+PIL.")

        # Compile regexes
        self._lot_prefix_re = re.compile("|".join(self.LOT_PREFIXES), re.IGNORECASE)
        self._exp_prefix_re = re.compile("|".join(self.EXP_PREFIXES), re.IGNORECASE)
        self._date_res = [re.compile(pat, re.IGNORECASE) for pat in self.DATE_PATTERNS]
        self._lot_code_re = re.compile(self.LOT_CODE_PATTERN, re.IGNORECASE)

    @staticmethod
    def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:  # grayscale
            return np.stack([frame, frame, frame], axis=-1)
        if frame.shape[2] == 4:
            return frame[..., :3]
        return frame

    def _ocr(self, frame: np.ndarray) -> List[OCRBox]:
        frame = self._ensure_rgb(frame)
        h, w = frame.shape[:2]
        results: List[OCRBox] = []
        if self._engine_name == "paddleocr":
            out = self._engine.ocr(frame)
            # out is a list per image; we pass one image so take out[0]
            if out[0] is None:
                return results
            for line in out[0]:
                if len(line) == 2:
                    # Old format: [bbox, (text, confidence)]
                    ((x1, y1), (x2, y2), (x3, y3), (x4, y4)), (txt, conf) = line
                elif len(line) == 3:
                    # New format: [bbox, (text, confidence), angle]
                    ((x1, y1), (x2, y2), (x3, y3), (x4, y4)), (txt, conf), _ = line
                else:
                    continue
                x_min = int(max(0, min(x1, x2, x3, x4)))
                y_min = int(max(0, min(y1, y2, y3, y4)))
                x_max = int(min(w - 1, max(x1, x2, x3, x4)))
                y_max = int(min(h - 1, max(y1, y2, y3, y4)))
                results.append(OCRBox(text=str(txt).strip(), conf=float(conf), box=(x_min, y_min, x_max, y_max)))
            return results

        elif self._engine_name == "tesseract":
            from pytesseract import Output  # lazy import
            pil = Image.fromarray(frame)
            data = pytesseract.image_to_data(pil, lang=self._tess_lang, output_type=Output.DICT)
            n = len(data["text"])
            for i in range(n):
                txt = data["text"][i].strip()
                if not txt:
                    continue
                conf = float(data["conf"][i]) if data["conf"][i].isdigit() else 0.0
                x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                results.append(OCRBox(text=txt, conf=conf, box=(x, y, x + bw, y + bh)))
            # Merge into line-level boxes
            results = self._merge_words_to_lines(results)
            return results
        else:
            return results

    @staticmethod
    def _merge_words_to_lines(words: List[OCRBox], y_tol: int = 8) -> List[OCRBox]:
        # Simple grouping by similar y center
        lines: List[List[OCRBox]] = []
        for w in words:
            cy = (w.box[1] + w.box[3]) / 2
            placed = False
            for line in lines:
                lcy = (line[0].box[1] + line[0].box[3]) / 2
                if abs(cy - lcy) <= y_tol:
                    line.append(w)
                    placed = True
                    break
            if not placed:
                lines.append([w])
        merged: List[OCRBox] = []
        for line in lines:
            line = sorted(line, key=lambda b: b.box[0])
            text = " ".join([b.text for b in line]).strip()
            conf = sum([b.conf for b in line]) / max(1, len(line))
            x1 = min([b.box[0] for b in line])
            y1 = min([b.box[1] for b in line])
            x2 = max([b.box[2] for b in line])
            y2 = max([b.box[3] for b in line])
            merged.append(OCRBox(text=text, conf=conf, box=(x1, y1, x2, y2)))
        return merged

    @staticmethod
    def _box_distance(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
        ax = (a[0] + a[2]) / 2; ay = (a[1] + a[3]) / 2
        bx = (b[0] + b[2]) / 2; by = (b[1] + b[3]) / 2
        return math.hypot(ax - bx, ay - by)

    def _find_dates(self, text: str) -> List[str]:
        found = []
        for dr in self._date_res:
            for m in dr.finditer(text):
                found.append(m.group(0))
        return found

    def _normalize_date(self, s: str) -> Optional[str]:
        s = s.strip().upper().replace(",", " ")
        # Try to coerce to YYYY-MM or YYYY-MM-DD where possible
        # 1) dd/mm/yyyy or dd-mm-yy etc.
        m = re.match(r"^([0-3]?\d)[/\-\. ]([0-1]?\d)[/\-\. ]((?:20)?\d{2})$", s)
        if m:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if y < 100: y += 2000
            try:
                if 1 <= mo <= 12 and 1 <= d <= 31:
                    return f"{y:04d}-{mo:02d}-{d:02d}"
            except Exception:
                return None
        # 2) mm/yyyy
        m = re.match(r"^([0-1]?\d)[/\-\. ]((?:20)?\d{2})$", s)
        if m:
            mo, y = int(m.group(1)), int(m.group(2))
            if y < 100: y += 2000
            if 1 <= mo <= 12:
                return f"{y:04d}-{mo:02d}"
        # 3) yyyy-mm
        m = re.match(r"^((?:20)?\d{2})[/\-\. ]([0-1]?\d)$", s)
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            if y < 100: y += 2000
            if 1 <= mo <= 12:
                return f"{y:04d}-{mo:02d}"
        # 4) Month words
        MONTHS = {
            # EN
            "JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"SEPT":9,"OCT":10,"NOV":11,"DEC":12,
            # ES
            "ENE":1,"ABR":4,"AGO":8,"DIC":12,
            # IT
            "GEN":1,"MAG":5,"GIU":6,"LUG":7,"SET":9,"OTT":10,"DIC":12,
            # DE/NL (mapped to closest)
            "MÄR":3,"MAJ":5,"OKT":10,"DEZ":12,
        }
        m = re.match(r"^([A-ZÄÖÜ]{3,4})[\.\- ]+((?:20)?\d{2})$", s)
        if m:
            mon = m.group(1)
            y = int(m.group(2));  y = y if y >= 1000 else (2000 + y)
            mon = mon.replace("Ä","A").replace("Ö","O").replace("Ü","U")
            if mon in MONTHS:
                return f"{y:04d}-{MONTHS[mon]:02d}"
        return None

    def _find_lot_code(self, text: str) -> Optional[str]:
        # Avoid lines that look like pure dates
        if any(dr.search(text) for dr in self._date_res):
            return None
        # Look for an alphanumeric code
        m = self._lot_code_re.search(text)
        if m:
            return m.group(1)
        return None

    def extract(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Returns:
        {
          "lot": FieldPrediction(...).__dict__,
          "expiry": FieldPrediction(...).__dict__,
          "engine": "paddleocr|tesseract",
          "lines": [ {text, conf, box}, ... ]
        }
        """
        boxes = self._ocr(frame)
        # Heuristics: locate prefix lines; search nearby for values.
        lot_candidates: List[Tuple[float, OCRBox, str]] = []  # (score, box, value)
        exp_candidates: List[Tuple[float, OCRBox, str]] = []

        # Pre-pass: collect any date-like values and lot-like codes from each line
        line_dates: Dict[int, List[str]] = {}
        line_lots: Dict[int, Optional[str]] = {}
        for i, b in enumerate(boxes):
            dates = self._find_dates(b.text)
            if dates:
                # normalize candidates
                norm_dates = [self._normalize_date(d) or d for d in dates]
                line_dates[i] = norm_dates
            else:
                line_dates[i] = []
            line_lots[i] = self._find_lot_code(b.text)

        # Pass: score candidates using prefixes and proximity
        for i, b in enumerate(boxes):
            txt = b.text.strip()
            has_lot_prefix = bool(self._lot_prefix_re.search(txt))
            has_exp_prefix = bool(self._exp_prefix_re.search(txt))

            # Nearby search radius in pixels (rough heuristic)
            def nearby(j):
                return self._box_distance(b.box, boxes[j].box) <= 200.0

            if has_lot_prefix:
                # Prefer lot-like codes on same or nearby lines
                if line_lots[i]:
                    lot_candidates.append((1.0 * b.conf, b, line_lots[i] or ""))
                for j in range(len(boxes)):
                    if j == i: continue
                    if nearby(j) and line_lots[j]:
                        # distance-weighted score
                        d = max(1e-3, self._box_distance(b.box, boxes[j].box))
                        score = b.conf * 0.6 + (boxes[j].conf * 0.4) + (150.0 / d)
                        lot_candidates.append((score, boxes[j], line_lots[j] or ""))

            if has_exp_prefix:
                # Prefer date-like values
                for dval in line_dates[i]:
                    exp_candidates.append((1.0 * b.conf, b, dval))
                for j in range(len(boxes)):
                    if j == i: continue
                    if nearby(j) and line_dates[j]:
                        d = max(1e-3, self._box_distance(b.box, boxes[j].box))
                        for dval in line_dates[j]:
                            score = b.conf * 0.6 + (boxes[j].conf * 0.4) + (150.0 / d)
                            exp_candidates.append((score, boxes[j], dval))

            # If no prefixes, still collect good generic candidates
            if not has_lot_prefix and line_lots[i]:
                lot_candidates.append((0.3 * b.conf, b, line_lots[i] or ""))
            if not has_exp_prefix and line_dates[i]:
                for dval in line_dates[i]:
                    exp_candidates.append((0.3 * b.conf, b, dval))

        # Pick best
        lot_pred = FieldPrediction(value=None, score=0.0, box=None, source_text=None)
        if lot_candidates:
            lot_candidates.sort(key=lambda t: t[0], reverse=True)
            best_score, best_box, best_val = lot_candidates[0]
            lot_pred = FieldPrediction(value=best_val, score=float(best_score), box=best_box.box, source_text=best_box.text)

        exp_pred = FieldPrediction(value=None, score=0.0, box=None, source_text=None)
        if exp_candidates:
            exp_candidates.sort(key=lambda t: t[0], reverse=True)
            best_score, best_box, best_val = exp_candidates[0]
            exp_pred = FieldPrediction(value=best_val, score=float(best_score), box=best_box.box, source_text=best_box.text)

        return {
            "lot": lot_pred.__dict__,
            "expiry": exp_pred.__dict__,
            "engine": self._engine_name,
            "lines": [b.__dict__ for b in boxes],
        }
