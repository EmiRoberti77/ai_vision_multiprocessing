import numpy as np
import cv2
import torch
import easyocr
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime
import re
import os
from app_base import AppBase
from db.db_logger import LoggerLevel
from db.error_codes import ErrorCode
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
try:
    from .ocr.common import clean_line, collapse_spaced_digits, find_lot_on_line, parse_expiry_from_text, has_exp_key
except Exception:
    try:
        from ray_actors.ocr.common import clean_line, collapse_spaced_digits, find_lot_on_line, parse_expiry_from_text, has_exp_key
    except Exception:
        from ocr.common import clean_line, collapse_spaced_digits, find_lot_on_line, parse_expiry_from_text, has_exp_key

class OCRProcessor(AppBase):
    """
    GPU-accelerated OCR processor for Ray actors
    Optimized for RTX 5090 with CUDA support
    """
    
    def __init__(self, channel_name:str, languages: List[str] = ['en'], gpu: bool = True):
        """
        Initialize OCR processor with optional GPU support (safe fallback to CPU)
        """
        self.gpu_requested = gpu
        self.gpu_available = False
        self.channel_name = channel_name
        # Only check CUDA if explicitly requested; guard against failures
        if gpu:
            try:
                self.gpu_available = bool(torch.cuda.is_available())
            except Exception as e:
                print(f"OCR Processor - CUDA check failed, forcing CPU: {e}")
                self.gpu_available = False
        self.device = "cuda" if self.gpu_available else "cpu"
        print(f"OCR Processor - Using device: {self.device}")
        
        # Initialize EasyOCR reader with fallback to CPU on any error
        try:
            self.reader = easyocr.Reader(languages, gpu=self.gpu_available)
        except Exception as e:
            msg = f"EasyOCR:init failed on {self.device} for {self.channel_name}, falling back to CPU: {e}"
            print(msg)
            self.app_logger.log_error(ErrorCode.OCR_ENGINE_FAILED, e)
            self.gpu_available = False
            self.device = "cpu"
            self.reader = easyocr.Reader(languages, gpu=False)
        
        # Best-effort device reporting
        try:
            if hasattr(self.reader, 'detector') and hasattr(self.reader, 'recognizer'):
                det_device = next(self.reader.detector.parameters()).device
                rec_device = next(self.reader.recognizer.parameters()).device
                print(f"OCR Detector device: {det_device}")
                print(f"OCR Recognizer device: {rec_device}")
        except Exception as e:
            print(f"Could not determine OCR model devices: {e}")
        
        # OCR processing parameters
        self.min_confidence = 0.35
        self.max_text_length = 100

    def save_ocr_frame(self, frame: np.ndarray) -> str:        
        # Get current timestamp
        now = datetime.now()
        timestamp = int(time.time())
        
        # Create directory structure: ROOT/ocr/year/month/day/hour/
        ocr_dir = os.path.join(
            ROOT, 
            "ocr", 
            str(now.year), 
            f"{now.month:02d}", 
            f"{now.day:02d}", 
            f"{now.hour:02d}"
        )
        
        # Create directories if they don't exist
        os.makedirs(ocr_dir, exist_ok=True)
        
        # Generate filename with timestamp and microseconds for uniqueness
        filename = f"ocr_frame_{timestamp}_{now.microsecond:06d}.jpg"
        full_path = os.path.join(ocr_dir, filename)
        
        # Save the frame as JPEG
        try:
            success = cv2.imwrite(full_path, frame)
            if success:
                print(f"OCR frame saved: {full_path=}")
                return full_path
            else:
                msg = f"EasyOCR:Failed to save OCR {self.channel_name=} frame: {full_path=}"
                print(msg)
                self.app_logger.log_error(ErrorCode.IMAGE_SAVE_FAILED, msg)
                return None
        except Exception as e:
            msg = f"Error saving OCR frame: {e}"
            print(msg)
            self.app_logger.log_error(ErrorCode.OCR_ENGINE_FAILED, msg)
            return None
        
    def preprocess_image(self, bgr_image: np.ndarray, rotate_90_clock: bool = True, save_ocr_images: bool = True) -> np.ndarray:

        if bgr_image is None or bgr_image.size == 0:
            return bgr_image
        
        # Rotate iPhone frames 90 degrees clockwise
        if rotate_90_clock:
            bgr_image = cv2.rotate(bgr_image, cv2.ROTATE_90_CLOCKWISE)
            
        h, w = bgr_image.shape[:2]
        
        # Resize if too small
        if max(h, w) < 320:
            scale = 1.5
            new_w = int(w * scale)
            new_h = int(h * scale)
            bgr_image = cv2.resize(bgr_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale and apply CLAHE
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        
        if save_ocr_images:
            try:
                # save images to disk for debugging
                now = datetime.now()
                ocr_run_dir=os.path.join(ROOT, 'ocr_run_process_image', self.channel_name, f"{now.year:02d}-{now.month:02d}-{now.day:02d}-{now.hour:02d}")
                os.makedirs(ocr_run_dir, exist_ok=True)
                cv2.imwrite(
                    os.path.join(ocr_run_dir,f"ocr_proc_img_original-{now.hour:02d}-{now.minute:02d}-{now.second:02d}-{now.microsecond:02d}.jpg"),
                    bgr_image
                )
                cv2.imwrite(
                    os.path.join(ocr_run_dir, f"ocr_proc_img_enhanced-{now.hour:02d}-{now.minute:02d}-{now.second:02d}-{now.microsecond:02d}.jpg"),
                    gray
                )
            except Exception as e:
                self.app_logger.log_error(ErrorCode.OCR_PROCESS_IMAGE_FAILED, str(e), self.channel_name)
        
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    
    def extract_text(self, bgr_image: np.ndarray, detail: int = 1, rotate_90_clock: bool = True) -> List[Tuple]:
        """
        Extract text from image using EasyOCR
        
        Args:
            bgr_image: Input BGR image
            detail: Detail level (0=text only, 1=text+bbox+confidence)
            rotate_iphone: Whether to rotate image 90 degrees clockwise for iPhone frames
            
        Returns:
            List of (bbox, text, confidence) tuples
        """
        if bgr_image is None or bgr_image.size == 0:
            self.app_logger.log_error(ErrorCode.INVALID_OCR_IMAGE_SIZE, 'Invalid image ssice - extract_text', self.channel_name)
            return []
        
        # Preprocess image with iPhone rotation
        processed = self.preprocess_image(bgr_image, rotate_90_clock=rotate_90_clock)
        
        try:
            # Perform OCR
            results = self.reader.readtext(processed, detail=detail, paragraph=False)
            
            # Filter results by confidence
            filtered_results = []
            for result in results:
                if detail == 1:
                    bbox, text, confidence = result
                    if confidence >= self.min_confidence and len(text.strip()) > 0:
                        filtered_results.append((bbox, text.strip(), confidence))
                else:
                    if len(result.strip()) > 0:
                        filtered_results.append(result.strip())
            
            return filtered_results
            
        except Exception as e:
            print(f"OCR extraction error: {e}")
            return []
    
    def process_frame(self, frame: np.ndarray, rotate_90_clock: bool = True, save_frame = True) -> Dict:
        """
        Process frame and extract OCR information
        
        Args:
            frame: Input frame
            rotate_iphone: Whether to rotate image 90 degrees clockwise for iPhone frames
            
        Returns:
            Dictionary with OCR results
        """
        start_time = time.time()
        
        # Extract text with bounding boxes and iPhone rotation
        ocr_results = self.extract_text(frame, detail=1, rotate_90_clock=rotate_90_clock)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Organize results
        text_lines = []
        all_text = []
        
        for bbox, text, confidence in ocr_results:
            text_lines.append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox
            })
            all_text.append(text)
        
        # Join all text
        full_text = " ".join(all_text)
        
        # Parse LOT and EXPIRY using robust spatial + lexical heuristics
        try:
            from ray_actors.ocr.text_parsing import parse_lot_and_expiry
        except Exception:
            try:
                from .ocr.text_parsing import parse_lot_and_expiry
            except Exception:
                from ocr.text_parsing import parse_lot_and_expiry

        lot, expiry = parse_lot_and_expiry(text_lines)

        if rotate_90_clock:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        ocr_image_path = self.save_ocr_frame(frame) if save_frame else "_EMPTY"
        
        return {
            'text': full_text,
            'lot': lot,
            'expiry': expiry,
            'lines': text_lines,
            'processing_time_ms': processing_time,
            'text_count': len(text_lines),
            'device': self.device,
            'ocr_image_path':ocr_image_path
        }
    
    def draw_ocr_results(self, frame: np.ndarray, ocr_results: Dict) -> np.ndarray:
        """
        Draw OCR results on frame with high visibility
        
        Args:
            frame: Input frame
            ocr_results: OCR results from process_frame
            
        Returns:
            Frame with OCR annotations
        """
        annotated_frame = frame.copy()
        
        # Draw bounding boxes and text
        for line in ocr_results.get('lines', []):
            bbox = line['bbox']
            text = line['text']
            confidence = line['confidence']
            
            # Convert bbox to integer coordinates
            bbox_int = np.array(bbox, dtype=np.int32)
            
            # Draw thick bounding box with bright blue color
            cv2.polylines(annotated_frame, [bbox_int], True, (255, 0, 0), 3)  # Bright blue, thick
            
            # Get top-left corner for text
            x, y = bbox_int[0]
            
            # Draw text with high contrast background
            text_with_conf = f"{text} ({confidence:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(text_with_conf, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(annotated_frame, 
                        (x, y - text_height - 15), 
                        (x + text_width + 10, y - 5), 
                        (0, 0, 0), -1)  # Black background
            
            # Draw white text on black background with outline
            cv2.putText(annotated_frame, text_with_conf, (x + 5, y - 10), 
                       font, font_scale, (0, 0, 0), thickness + 1)  # Black outline
            cv2.putText(annotated_frame, text_with_conf, (x + 5, y - 10), 
                       font, font_scale, (255, 255, 255), thickness)  # White text
        
        # Add processing info with high contrast
        info_text = f"OCR: {ocr_results.get('text_count', 0)} texts, {ocr_results.get('processing_time_ms', 0):.1f}ms"
        
        # Draw background for info text
        (info_width, info_height), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated_frame, (5, 5), (info_width + 15, info_height + 15), (0, 0, 0), -1)
        
        # Draw info text
        cv2.putText(annotated_frame, info_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame
    
    def get_memory_usage(self) -> Dict:
        """
        Get GPU memory usage for monitoring
        
        Returns:
            Dictionary with memory information
        """
        if self.gpu_available:
            return {
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                'device': str(torch.cuda.current_device())
            }
        else:
            return {
                'gpu_memory_allocated_mb': 0,
                'gpu_memory_reserved_mb': 0,
                'device': 'cpu'
            }
