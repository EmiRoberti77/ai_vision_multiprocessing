import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from app_base import AppBase
from db.error_codes import ErrorCode

# PaddleOCR imports with graceful fallback
try:
    from paddleocr import PaddleOCR
    import paddle
except Exception as e:  # pragma: no cover - import-time guard for environments without Paddle
    PaddleOCR = None
    paddle = None

from ocr.text_parsing import parse_lot_and_expiry


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class OCRProcessorPaddle(AppBase):
    """
    PaddleOCR-based OCR processor with the same public API as the EasyOCR version.
    - Accepts the same constructor signature
    - Exposes: save_ocr_frame, preprocess_image, extract_text, process_frame,
               draw_ocr_results, get_memory_usage
    """

    def __init__(self, channel_name: str, languages: List[str] = ['en'], gpu: bool = True):
        super().__init__()
        self.channel_name = channel_name
        self.gpu_requested = bool(gpu)
        self.gpu_available = False
        self.device = 'cpu'

        # Determine GPU availability via Paddle (if installed)
        if self.gpu_requested and paddle is not None:
            try:
                self.gpu_available = bool(paddle.is_compiled_with_cuda())
            except Exception:
                self.gpu_available = False

        self.device = 'cuda' if self.gpu_available else 'cpu'
        print(f"PaddleOCR - Using device: {self.device}")

        # Initialize PaddleOCR engine
        lang_code = languages[0] if isinstance(languages, list) and len(languages) > 0 else 'en'

        if PaddleOCR is None:
            msg = "PaddleOCR not installed. Please install 'paddlepaddle-gpu' and 'paddleocr'."
            print(msg)
            self.app_logger.log_error(ErrorCode.OCR_ENGINE_FAILED, msg)
            # Defer raising to allow caller to fallback
            raise ImportError(msg)

        try:
            self.reader = PaddleOCR(
                use_angle_cls=True,
                lang=lang_code,
                use_gpu=self.gpu_available,
                show_log=False,
                det=True,
                rec=True
            )
        except Exception as e:
            msg = f"PaddleOCR:init failed on {self.device} for {self.channel_name}: {e}"
            print(msg)
            self.app_logger.log_error(ErrorCode.OCR_ENGINE_FAILED, msg)
            # Force CPU retry once
            try:
                self.gpu_available = False
                self.device = 'cpu'
                self.reader = PaddleOCR(
                    use_angle_cls=True,
                    lang=lang_code,
                    use_gpu=False,
                    show_log=False,
                    det=True,
                    rec=True
                )
            except Exception as e2:
                msg2 = f"PaddleOCR:init failed on CPU for {self.channel_name}: {e2}"
                print(msg2)
                self.app_logger.log_error(ErrorCode.OCR_ENGINE_FAILED, msg2)
                raise

        # Parameters
        self.min_confidence = 0.5
        self.max_text_length = 120

    def save_ocr_frame(self, frame: np.ndarray) -> str:
        now = datetime.now()
        timestamp = int(time.time())
        ocr_dir = os.path.join(
            ROOT,
            "ocr",
            str(now.year),
            f"{now.month:02d}",
            f"{now.day:02d}",
            f"{now.hour:02d}"
        )
        os.makedirs(ocr_dir, exist_ok=True)
        filename = f"ocr_frame_{timestamp}_{now.microsecond:06d}.jpg"
        full_path = os.path.join(ocr_dir, filename)
        try:
            ok = cv2.imwrite(full_path, frame)
            if ok:
                print(f"OCR frame saved: full_path={full_path}")
                return full_path
            else:
                msg = f"PaddleOCR:Failed to save OCR channel={self.channel_name} frame: full_path={full_path}"
                print(msg)
                self.app_logger.log_error(ErrorCode.IMAGE_SAVE_FAILED, msg)
                return None
        except Exception as e:
            msg = f"Error saving OCR frame: {e}"
            print(msg)
            self.app_logger.log_error(ErrorCode.IMAGE_SAVE_FAILED, msg)
            return None

    def preprocess_image(self, bgr_image: np.ndarray, rotate_iphone: bool = True) -> np.ndarray:
        if bgr_image is None or getattr(bgr_image, 'size', 0) == 0:
            self.app_logger.log_error(ErrorCode.OCR_PROCESS_IMAGE_FAILED, f"process_image()-bgr_image=None-channel={self.channel_name}", "image covertion error")
            return bgr_image
        img = bgr_image
        if rotate_iphone:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]
        if max(h, w) < 320:
            scale = 1.5
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        now = datetime.now()
        try:
            cv2.imwrite(
                f"ocr_process_image-{now.year:02d}{now.month:02d}-{now.day:02d}-{now.hour:02d}-{now.minute:02d}-{now.second:02d}-{now.microsecond:02d}.jpg",
                img
            )
        except Exception:
            pass
        return img

    def extract_text(self, bgr_image: np.ndarray, detail: int = 1, rotate_iphone: bool = True) -> Tuple[List[Tuple], np.ndarray]:
        """
        Returns a tuple of (ocr_tuples, processed_frame), where ocr_tuples is a list of
        (bbox, text, confidence) and processed_frame is the preprocessed image used for OCR.
        """
        if bgr_image is None or getattr(bgr_image, 'size', 0) == 0:
            print(f"WARN:extract frame no zise")
            return [], bgr_image

        processed = self.preprocess_image(bgr_image, rotate_iphone=rotate_iphone)

        # PaddleOCR accepts numpy arrays (BGR). We keep as-is.
        try:
            results = self.reader.ocr(processed, cls=True)
        except Exception as e:
            self.app_logger.log_error(ErrorCode.OCR_ENGINE_FAILED, str(e))
            print(f"ERROR:PaddleOCR extraction error: {e}")
            return [], processed

        # results is a list per image; for ndarray input it's typically a nested list
        # Normalize into list of (bbox, text, confidence)
        normalized: List[Tuple[List[List[float]], str, float]] = []
        try:
            batches = results if isinstance(results, list) else [results]
            for batch in batches:
                # Each batch may be None or list of lines
                if batch is None:
                    continue
                for item in batch:
                    # item formats observed:
                    # 1) [bbox, (text, conf)]
                    # 2) { 'text':..., 'confidence':..., 'bbox':... } (less common)
                    bbox = None
                    text = ''
                    conf = 0.0
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        bbox = item[0]
                        rec = item[1]
                        if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                            text = str(rec[0])
                            try:
                                conf = float(rec[1])
                            except Exception:
                                conf = 0.0
                    elif isinstance(item, dict):
                        bbox = item.get('bbox')
                        text = str(item.get('text', ''))
                        try:
                            conf = float(item.get('confidence', 0.0))
                        except Exception:
                            conf = 0.0

                    if bbox is None:
                        continue
                    if text and conf >= self.min_confidence:
                        normalized.append((bbox, text.strip(), conf))
        except Exception as e:
            print(f"PaddleOCR normalization error: {e}")
            return [], processed

        return normalized, processed

    def process_frame(self, frame: np.ndarray, rotate_90_clock: bool = True, save_frame = True) -> Dict:
        t0 = time.time()
        print(f"process_frame {rotate_90_clock=}")
        print(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        ocr_out = self.extract_text(frame, detail=1, rotate_iphone=rotate_90_clock)
        ocr_results, processed_frame = ocr_out
        print(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # Backward/defensive handling: support either list-only or (list, processed_frame)
        # if isinstance(ocr_out, tuple) and len(ocr_out) == 2:
        #     ocr_results, processed_frame = ocr_out
        # else:
        #     ocr_results = ocr_out  # type: ignore[assignment]
        #     processed_frame = self.preprocess_image(frame, rotate_iphone=rotate_iphone)
        proc_ms = (time.time() - t0) * 1000.0

        text_lines: List[Dict] = []
        all_text: List[str] = []
        for bbox, text, conf in ocr_results:
            text_lines.append({
                'text': text,
                'confidence': float(conf),
                'bbox': bbox
            })
            all_text.append(text)

        full_text = " ".join(all_text)

        lot, expiry = parse_lot_and_expiry(text_lines)

        ocr_image_path = self.save_ocr_frame(processed_frame) if save_frame else "_EMPTY"

        return {
            'text': full_text,
            'lot': lot,
            'expiry': expiry,
            'lines': text_lines,
            'processing_time_ms': proc_ms,
            'text_count': len(text_lines),
            'device': self.device,
            'ocr_image_path': ocr_image_path
        }

    def draw_ocr_results(self, frame: np.ndarray, ocr_results: Dict) -> np.ndarray:
        annotated = frame.copy()
        for line in ocr_results.get('lines', []):
            bbox = line.get('bbox')
            text = line.get('text', '')
            conf = float(line.get('confidence', 0.0))
            try:
                bbox_int = np.array(bbox, dtype=np.int32)
            except Exception:
                continue
            cv2.polylines(annotated, [bbox_int], True, (255, 0, 0), 3)
            x, y = bbox_int[0]
            txt = f"{text} ({conf:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(txt, font, font_scale, thickness)
            cv2.rectangle(annotated, (x, y - th - 15), (x + tw + 10, y - 5), (0, 0, 0), -1)
            cv2.putText(annotated, txt, (x + 5, y - 10), font, font_scale, (0, 0, 0), thickness + 1)
            cv2.putText(annotated, txt, (x + 5, y - 10), font, font_scale, (255, 255, 255), thickness)
        info = f"OCR: {ocr_results.get('text_count', 0)} texts, {ocr_results.get('processing_time_ms', 0.0):.1f}ms"
        (iw, ih), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (5, 5), (iw + 15, ih + 15), (0, 0, 0), -1)
        cv2.putText(annotated, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return annotated

    def get_memory_usage(self) -> Dict:
        if self.gpu_available and paddle is not None:
            # Paddle does not expose per-model memory like torch; return device info
            try:
                place = paddle.device.get_device()
            except Exception:
                place = 'gpu'
            return {
                'gpu_memory_allocated_mb': 0,
                'gpu_memory_reserved_mb': 0,
                'device': str(place)
            }
        else:
            return {
                'gpu_memory_allocated_mb': 0,
                'gpu_memory_reserved_mb': 0,
                'device': 'cpu'
            }


