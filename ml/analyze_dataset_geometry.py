import os
import sys
import glob
import numpy as np
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.config import TRAIN_DIR, CLASSES

def analyze_dataset_sign_geometry():
    print("=" * 80)
    print("ANALYZING ASL DATASET SIGN GEOMETRY & ORIENTATION")
    print("=" * 80)

    for sign in ['A', 'B', 'C', 'D', 'G', 'H', 'L', 'O', 'Y', 'del', 'nothing', 'space']:
        sample_path = os.path.join(TRAIN_DIR, sign, f"{sign}1.jpg")
        img = Image.open(sample_path)
        arr = np.array(img)
        # Background is usually darker or lighter; let's check mean brightness and dimensions
        print(f"Sign: {sign:<8} | Shape: {arr.shape} | RGB Mean: {arr.mean(axis=(0,1)).round(1)} | Min: {arr.min()}, Max: {arr.max()}")

if __name__ == "__main__":
    analyze_dataset_sign_geometry()
