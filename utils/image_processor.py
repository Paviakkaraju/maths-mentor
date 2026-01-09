import easyocr
import cv2
import numpy as np
from PIL import Image
import io

class ImageProcessor:
    """
    Handles Image-to-Text conversion using EasyOCR.
    Optimized for extracting math problems from photos/screenshots.
    """
    def __init__(self, languages=['en']):
        # Initialize the reader (downloads models on first run)
        # gpu=False if you don't have a dedicated GPU
        self.reader = easyocr.Reader(languages, gpu=False)

    def _preprocess_image(self, image_bytes):
        """
        Applies basic image processing to improve OCR accuracy.
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 1. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Denoising
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # 3. Thresholding (Optional: helps if the background is messy)
        # _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return denoised

    def extract_text(self, image_bytes) -> dict:
        """
        Performs OCR and returns structured results.
        """
        try:
            # Preprocess
            processed_img = self._preprocess_image(image_bytes)
            
            # Perform OCR
            # detail=0 would return only text, but we need confidence scores
            results = self.reader.readtext(processed_img)
            
            full_text = []
            confidences = []
            
            for (bbox, text, prob) in results:
                full_text.append(text)
                confidences.append(prob)
            
            # Combine text and calculate average confidence
            final_text = " ".join(full_text)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "extracted_text": final_text,
                "confidence": round(avg_confidence, 2),
                "success": True
            }
            
        except Exception as e:
            return {
                "extracted_text": "",
                "confidence": 0,
                "success": False,
                "error": str(e)
            }

# --- Backend Test Script ---
if __name__ == "__main__":
    processor = ImageProcessor()
    
    # Test with a dummy image path
    with open("image.png", "rb") as f:
        image_data = f.read()
        result = processor.extract_text(image_data)
        print(f"Extracted: {result['extracted_text']}")
        print(f"Confidence: {result['confidence']}")