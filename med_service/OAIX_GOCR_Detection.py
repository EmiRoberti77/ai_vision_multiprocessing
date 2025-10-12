import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import os
import json
load_dotenv() 

class OAIX_GOCR_Detection():
    def __init__(self) -> None:
        if not os.getenv('OAIX_G_OCR'):
            raise ValueError("Error:Missing OAIX_G_OCR")

        # Configure the Gemini API
        genai.configure(api_key=os.getenv('OAIX_G_OCR'))
        
        # Create the model with system instruction
        self.model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.1,  # Lower temperature for more consistent, faster responses
                # "max_output_tokens": 100,  # Limit output tokens for faster processing
                # "candidate_count": 1  # Only generate one candidate
            },
            system_instruction=(
                "You are an expert data extractor. Your task is to analyze the provided "
                "medicine label image and extract the Lot/Batch number and the Expiry Date. "
                "You must return the result as a single JSON object. "
                "Use 'lot_number' and 'expiry_date' as the keys. "
                "Format the expiry_date in YYYY-MM-DD format if possible."
            )
        )
    
    def gem_detect(self, image):
        img = Image.open(image)
        
        prompt = "Extract the Lot/Batch number and Expiry Date from this medicine package. Return as JSON with keys 'lot_number' and 'expiry_date'."
        
        response = self.model.generate_content([prompt, img])
        
        try:
            extracted_data = json.loads(response.text)
            return extracted_data
        except json.JSONDecodeError:
            print("Error: OAIXGOCR did not return valid JSON.")
            print("Raw response:", response.text)
            return None