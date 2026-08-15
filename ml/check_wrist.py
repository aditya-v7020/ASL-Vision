import os
import sys
import numpy as np
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.config import TRAIN_DIR, CLASSES

def check_wrist_entry_side():
    print("=" * 80)
    print("ANALYZING WRIST / FOREARM ENTRY POINT IN DATASET")
    print("=" * 80)

    for sign in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'O', 'V', 'W', 'Y']:
        path = os.path.join(TRAIN_DIR, sign, f"{sign}1.jpg")
        img = Image.open(path).convert('RGB')
        arr = np.array(img)
        # Look at bottom 20 rows of the 200x200 image (rows 180 to 200)
        bottom_strip = arr[180:, :, :]
        # Check mean brightness across columns 0-100 (left) vs 100-200 (right)
        left_val = bottom_strip[:, :100, :].mean()
        right_val = bottom_strip[:, 100:, :].mean()
        
        # Look at right 20 columns (cols 180-200) vs left 20 columns (cols 0-20)
        left_edge = arr[:, :20, :].mean()
        right_edge = arr[:, 180:, :].mean()
        
        print(f"Sign: {sign:<4} | Bottom-Left: {left_val:5.1f} | Bottom-Right: {right_val:5.1f} | Left-Edge: {left_edge:5.1f} | Right-Edge: {right_edge:5.1f}")

if __name__ == "__main__":
    check_wrist_entry_side()
