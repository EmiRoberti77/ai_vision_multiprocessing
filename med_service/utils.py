from datetime import datetime
import os
import cv2
from PIL import Image
import numpy as np

def file_name(prefix:str, ext:str='.jpg')->str:
    now = datetime.now()
    return f"{prefix}-{now.year}-{now.month}-{now.day}T{now.hour}-{now.minute}-{now.second}{ext}"


def folder_name(prefix:str)->str:
    now = datetime.now()
    return f"{prefix}-{now.year}-{now.month}-{now.day}T{now.hour}-{now.minute}-{now.second}"


def create_run_folder_output(save_run_root:str, prefix:str)->str:
    save_run_path = os.path.join(save_run_root, folder_name(prefix=prefix))
    if not os.path.exists(save_run_path):
            os.makedirs(save_run_path)
    return save_run_path

def fmt_seconds(s):
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:05.2f}"


def optimize_image_for_ocr(image_path: str, max_size: int = 800) -> str:
    """Resize and optimize image for faster OCR - reduces 24MB to ~200KB"""
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Calculate scaling to fit within max_size while maintaining aspect ratio
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img
    
    # Save optimized version with high compression
    optimized_path = image_path.replace('.jpg', '_opt.jpg')
    cv2.imwrite(optimized_path, img_resized, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return optimized_path

def optimize_roi_in_memory(roi_image: np.ndarray, max_size: int = 400) -> Image.Image:
    """Process ROI directly in memory without file I/O"""
    height, width = roi_image.shape[:2]
    
    # Resize
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_resized = cv2.resize(roi_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        img_resized = roi_image
    
    # Convert to grayscale and sharpen
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img_sharp = cv2.filter2D(img_gray, -1, kernel)
    
    # Convert to PIL Image directly
    return Image.fromarray(img_sharp)
