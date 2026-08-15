import os
import sys
import glob
import numpy as np
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.config import TRAIN_DIR, TEST_DIR, CLASSES, IMAGE_SIZE
from backend.predictor import predictor

def inspect_dataset_samples():
    print("=" * 80)
    print("STEP 1: INSPECTING DATASET IMAGES & PREDICTOR")
    print("=" * 80)

    predictor.load_model()

    test_signs = ['A', 'B', 'C', 'D', 'G', 'H', 'L', 'O', 'Y', 'del', 'nothing', 'space']
    
    print(f"{'Class':<10} | {'File':<15} | {'Image Size':<12} | {'Mode':<6} | {'Predicted':<10} | {'Confidence':<10} | {'Top-3 Breakdown'}")
    print("-" * 100)

    for sign in test_signs:
        sample_path = os.path.join(TRAIN_DIR, sign, f"{sign}100.jpg")
        if not os.path.exists(sample_path):
            sample_path = glob.glob(os.path.join(TRAIN_DIR, sign, "*.jpg"))[0]
        
        img = Image.open(sample_path)
        pred, conf, top3 = predictor.predict_pil_image(img)
        top3_str = ", ".join([f"{p['class']}:{p['confidence']}%" for p in top3])
        print(f"{sign:<10} | {os.path.basename(sample_path):<15} | {str(img.size):<12} | {img.mode:<6} | {pred:<10} | {conf:<10.2f} | {top3_str}")

if __name__ == "__main__":
    inspect_dataset_samples()
