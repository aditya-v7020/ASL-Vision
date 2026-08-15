import os
import sys
import numpy as np
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.predictor import predictor
from ml.config import TRAIN_DIR

def test_original_vs_flipped():
    predictor.load_model()
    test_signs = ['A', 'B', 'C', 'D', 'G', 'H', 'L', 'O', 'Y', 'del', 'nothing', 'space']
    
    print("=" * 100)
    print(f"{'Class':<8} | {'ORIGINAL Predicted':<18} | {'Conf':<8} | {'FLIPPED Predicted':<18} | {'Conf':<8}")
    print("-" * 100)

    for sign in test_signs:
        path = os.path.join(TRAIN_DIR, sign, f"{sign}1.jpg")
        img = Image.open(path).convert('RGB')
        
        # Original
        pred_orig = predictor.predict_image_bytes(open(path, 'rb').read())
        
        # Flipped
        img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        import io
        buf = io.BytesIO()
        img_flipped.save(buf, format='JPEG')
        pred_flip = predictor.predict_image_bytes(buf.getvalue())
        
        print(f"{sign:<8} | {pred_orig['prediction']:<18} | {pred_orig['confidence']:6.2f}% | {pred_flip['prediction']:<18} | {pred_flip['confidence']:6.2f}%")

if __name__ == "__main__":
    test_original_vs_flipped()
