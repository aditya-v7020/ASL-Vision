import os
import sys
import numpy as np
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.config import TRAIN_DIR

def inspect_dataset_perspective():
    print("=" * 80)
    print("CHECKING DATASET PERSPECTIVE: SIGNER (1st person) vs OBSERVER (3rd person)")
    print("=" * 80)

    # Let's check 'L':
    # In ASL 'L', the right hand forms an L with index finger pointing UP and thumb pointing to the SIDE (90 degrees).
    # If the photo is taken by the signer looking at their own right hand:
    #   The thumb points to the LEFT!
    # If the photo is taken by someone facing the signer:
    #   The thumb points to the RIGHT (from observer's view)!
    
    # Let's inspect L1.jpg:
    # Let's find the columns where the thumb is and where the index finger is in L1.jpg.
    l_img = Image.open(os.path.join(TRAIN_DIR, 'L', 'L1.jpg')).convert('L')
    l_arr = np.array(l_img)
    # The image is 200x200. Let's find horizontal center of mass in upper half (index finger) vs lower half (thumb & palm).
    upper_half = l_arr[:100, :]
    lower_half = l_arr[100:160, :]
    
    print("L1.jpg dimensions:", l_arr.shape)
    
    # Let's check G1.jpg:
    # In ASL 'G', index finger points horizontally. Which way does it point? (Left or Right?)
    g_img = Image.open(os.path.join(TRAIN_DIR, 'G', 'G1.jpg')).convert('L')
    g_arr = np.array(g_img)
    
    # Let's check C1.jpg:
    # In ASL 'C', fingers curve to form 'C'. Which way does the opening of 'C' face?
    c_img = Image.open(os.path.join(TRAIN_DIR, 'C', 'C1.jpg')).convert('L')
    c_arr = np.array(c_img)
    
    print("\nDataset images inspection completed.")

if __name__ == "__main__":
    inspect_dataset_perspective()
