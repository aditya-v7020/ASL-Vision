import os
import sys
import glob
from PIL import Image
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.predictor import predictor
from ml.config import TRAIN_DIR, CLASSES

def test_webcam_pipeline_sim():
    print("=" * 80)
    print("SIMULATING WEBCAM CAPTURE & CROP PIPELINE")
    print("=" * 80)
    
    predictor.load_model()
    
    # We will test dataset sample images placed inside a 640x480 simulated webcam frame,
    # then cropped via the exact JS crop math, and evaluated.
    test_signs = ['A', 'B', 'C', 'D', 'G', 'H', 'L', 'O', 'Y', 'del', 'nothing', 'space']
    
    for sign in test_signs:
        sample_path = os.path.join(TRAIN_DIR, sign, f"{sign}10.jpg")
        img = Image.open(sample_path).convert('RGB') # 200x200
        
        # Paste onto 640x480 neutral canvas (simulating camera feed where hand is in center box)
        feed = Image.new('RGB', (640, 480), (128, 128, 128))
        # Center the 200x200 hand inside the 360x360 center crop area
        resized_hand = img.resize((320, 320))
        feed.paste(resized_hand, (160, 80))
        
        # Now apply the JS crop:
        videoWidth, videoHeight = 640, 480
        minDimension = min(videoWidth, videoHeight)
        cropSize = int(minDimension * 0.75) # 360
        startX = int((videoWidth - cropSize) / 2) # 140
        startY = int((videoHeight - cropSize) / 2) # 60
        
        cropped = feed.crop((startX, startY, startX + cropSize, startY + cropSize)).resize((200, 200))
        
        import io
        buf = io.BytesIO()
        cropped.save(buf, format='JPEG', quality=90)
        res = predictor.predict_image_bytes(buf.getvalue())
        
        top3_str = ", ".join([f"{p['class']}:{p['confidence']}%" for p in res['top_predictions']])
        match = "[MATCH]" if res['prediction'] == sign else "[DIFF]"
        print(f"Sign: {sign:<8} | Predicted: {res['prediction']:<8} | Conf: {res['confidence']:6.2f}% | Top-3: {top3_str} {match}")

if __name__ == "__main__":
    test_webcam_pipeline_sim()
