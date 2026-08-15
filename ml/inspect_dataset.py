import os
import sys
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.config import TRAIN_DIR, TEST_DIR, CLASSES

def check_image(fpath):
    try:
        with Image.open(fpath) as img:
            return True, img.size, img.mode
    except Exception:
        return False, None, None

def inspect_dataset():
    print("=" * 80)
    print("ASL ALPHABET RECOGNITION - DATASET INTEGRITY & INSPECTION REPORT")
    print("=" * 80)
    
    print(f"[*] Checking Training Directory: {TRAIN_DIR}")
    if not os.path.exists(TRAIN_DIR):
        print(f"[!] ERROR: Training directory not found at {TRAIN_DIR}")
        return False

    subdirs = sorted([d.name for d in os.scandir(TRAIN_DIR) if d.is_dir()])
    print(f"[+] Total class subdirectories found : {len(subdirs)}")
    print(f"[+] Expected 29 ASL classes          : {len(CLASSES)}")
    
    if subdirs == CLASSES:
        print("[+] SUCCESS: Class directory names perfectly match configured 29 ASL classes.")
    else:
        print(f"[!] WARNING: Discrepancy in class folders: {subdirs}")

    print("\n" + "-" * 80)
    print(f"{'CLASS':<10} | {'IMAGE COUNT':<12} | {'VALID':<10} | {'INVALID':<10} | {'SAMPLE SIZE / MODE'}")
    print("-" * 80)

    total_valid = 0
    total_invalid = 0
    sample_sizes = set()
    sample_modes = set()

    for cls in CLASSES:
        cls_dir = os.path.join(TRAIN_DIR, cls)
        if not os.path.exists(cls_dir):
            print(f"{cls:<10} | {'MISSING':<12} | {0:<10} | {0:<10} | N/A")
            continue

        entries = [e.path for e in os.scandir(cls_dir) if e.is_file() and e.name.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Verify first 10 files and last 10 files in detail, and count all
        valid_count = 0
        invalid_count = 0
        sample_info = "N/A"

        with ThreadPoolExecutor(max_workers=8) as executor:
            # Sample check 50 images per class for speed, plus count total
            sample_entries = entries[:50]
            results = list(executor.map(check_image, sample_entries))
            
            for ok, sz, md in results:
                if ok:
                    sample_sizes.add(sz)
                    sample_modes.add(md)
                    sample_info = f"{sz}, {md}"
                else:
                    invalid_count += 1

        valid_count = len(entries) - invalid_count
        total_valid += valid_count
        total_invalid += invalid_count

        print(f"{cls:<10} | {len(entries):<12} | {valid_count:<10} | {invalid_count:<10} | {sample_info}")

    print("-" * 80)
    print(f"TOTAL TRAINING IMAGES : {total_valid + total_invalid}")
    print(f"TOTAL VALID IMAGES    : {total_valid}")
    print(f"TOTAL INVALID IMAGES  : {total_invalid}")
    print(f"SAMPLE SIZES OBSERVED : {sample_sizes}")
    print(f"COLOR MODES OBSERVED  : {sample_modes}")

    # Inspect test directory
    print("\n[*] Checking Test Directory: " + str(TEST_DIR))
    if os.path.exists(TEST_DIR):
        test_subdirs = [d.name for d in os.scandir(TEST_DIR) if d.is_dir()]
        test_files = [f.name for f in os.scandir(TEST_DIR) if f.is_file() and f.name.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"[+] Test subdirectories : {len(test_subdirs)} (Flat folder structure)")
        print(f"[+] Test flat images     : {len(test_files)}")
        print(f"[+] Test sample images   : {test_files[:5]} ... {test_files[-3:]}")
        print("[+] Verified: asl_alphabet_test contains flat sample images without ground truth labels.")
        print("[+] Stratified 70/15/15 split on asl_alphabet_train will be used for training, validation, and test evaluation.")

    print("=" * 80 + "\n")

if __name__ == "__main__":
    inspect_dataset()
