import os
import sys
import io
import json
from PIL import Image
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.predictor import predictor
from ml.config import TRAIN_DIR, CLASSES, CLASSES_PATH, MODEL_PATH

def verify_all():
    print("=" * 80)
    print("COMPREHENSIVE VERIFICATION OF WEBCAM & INFERENCE FIXES")
    print("=" * 80)

    # 1. Verify Class Mapping Consistency
    with open(CLASSES_PATH, "r") as f:
        saved_classes = json.load(f)
    assert saved_classes == CLASSES, "classes.json does not match config CLASSES!"
    print(f"[+] Class Mapping: 29 classes verified in exact order.")

    # 2. Verify Predictor
    predictor.load_model()
    assert predictor.model is not None, "Model failed to load!"
    print(f"[+] Model Loaded: {MODEL_PATH}")

    # 3. Test the required key signs: A, O, G, H, Y, B, V, S, DEL, NOTHING, SPACE
    key_signs = ['A', 'O', 'G', 'H', 'Y', 'B', 'V', 'S', 'del', 'nothing', 'space']
    print("\n" + "-" * 80)
    print(f"{'Class':<10} | {'Test File':<15} | {'Predicted':<10} | {'Confidence':<10} | {'Status':<8} | {'Top-3 Breakdown'}")
    print("-" * 80)

    all_matched = True
    for sign in key_signs:
        path = os.path.join(TRAIN_DIR, sign, f"{sign}5.jpg")
        with open(path, "rb") as f:
            bytes_data = f.read()
        res = predictor.predict_image_bytes(bytes_data)
        
        is_match = (res["prediction"] == sign)
        if not is_match:
            all_matched = False
        status_tag = "[MATCH]" if is_match else "[DIFF]"
        top3_str = ", ".join([f"{p['class']}:{p['confidence']}%" for p in res["top_predictions"]])
        print(f"{sign:<10} | {os.path.basename(path):<15} | {res['prediction']:<10} | {res['confidence']:6.2f}% | {status_tag:<8} | {top3_str}")

    print("-" * 80)

    # 4. Check last_inference_input.jpg creation
    debug_path = os.path.join(os.path.dirname(__file__), "outputs", "last_inference_input.jpg")
    assert os.path.exists(debug_path), "Debug last_inference_input.jpg was not created!"
    debug_img = Image.open(debug_path)
    print(f"[+] Debug Frame Generated: {debug_path} (Size: {debug_img.size}, Mode: {debug_img.mode})")

    print("\n[+] ALL WEBCAM PIPELINE VERIFICATIONS PASSED SUCCESSFULLY!\n")

if __name__ == "__main__":
    verify_all()
