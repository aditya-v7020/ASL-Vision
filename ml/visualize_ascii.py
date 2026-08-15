import os
import sys
import numpy as np
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.config import TRAIN_DIR

def inspect_sign_pixels(sign_name, file_name):
    path = os.path.join(TRAIN_DIR, sign_name, file_name)
    img = Image.open(path).convert('L').resize((20, 20))
    arr = np.array(img)
    threshold = (arr.max() + arr.min()) / 2
    binary = np.where(arr > threshold, '#', ' ')
    print(f"\nSign: {sign_name} ({file_name}) (20x20 thumbnail):")
    for row in binary:
        print("".join(row))

if __name__ == "__main__":
    for sign in ['A', 'B', 'C', 'D', 'G', 'H', 'L', 'O', 'Y']:
        inspect_sign_pixels(sign, f"{sign}1.jpg")
