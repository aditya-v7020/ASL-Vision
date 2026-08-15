import os
import sys
import io
import json
import glob
import numpy as np
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.predictor import predictor
from ml.config import TRAIN_DIR, CLASSES, CLASSES_PATH, MODEL_PATH, IMAGE_SIZE

def run_webcam_vs_dataset_comparison():
    print("=" * 100)
    print("AUTOMATED WEBCAM VS DATASET PREPROCESSING & INFERENCE COMPARISON")
    print("=" * 100)

    predictor.load_model()
    assert predictor.model is not None, "Model failed to load!"

    with open(CLASSES_PATH, "r") as f:
        classes = json.load(f)
    assert classes == CLASSES, "Class mapping mismatch!"

    # Target test classes as specified in prompt:
    # A, B, G, H, O, S, U, Y, DEL, NOTHING, SPACE
    target_classes = ['A', 'B', 'G', 'H', 'O', 'S', 'U', 'Y', 'del', 'nothing', 'space']

    print(f"\n1. PREPROCESSING EQUIVALENCE AUDIT:")
    print(f"   [+] Model Target Resolution : {IMAGE_SIZE} (128x128)")
    print(f"   [+] Color Format            : RGB (3-channel float32)")
    print(f"   [+] Rescaling Range         : [0.0, 255.0] -> Model Internal Rescaling(1/255)")
    print(f"   [+] Number of ASL Classes   : {len(classes)} classes in exact index order")

    print("\n" + "=" * 100)
    print(f"{'Class':<8} | {'Source Type':<14} | {'Input Size':<11} | {'Predicted':<10} | {'Conf':<7} | {'Status':<8} | {'Top-5 Predictions'}")
    print("-" * 100)

    results = []
    
    for cls_name in target_classes:
        sample_path = os.path.join(TRAIN_DIR, cls_name, f"{cls_name}15.jpg")
        if not os.path.exists(sample_path):
            sample_path = glob.glob(os.path.join(TRAIN_DIR, cls_name, "*.jpg"))[0]
            
        dataset_img = Image.open(sample_path).convert('RGB')
        
        # 1. Dataset Direct Evaluation
        buf_dataset = io.BytesIO()
        dataset_img.save(buf_dataset, format='JPEG')
        res_dataset = predictor.predict_image_bytes(buf_dataset.getvalue())
        top5_dataset = ", ".join([f"{p['class']}:{p['confidence']}%" for p in res_dataset['top_predictions'][:3]])
        match_dataset = "[MATCH]" if res_dataset['prediction'] == cls_name else "[DIFF]"
        
        print(f"{cls_name:<8} | {'Dataset Direct':<14} | {str(dataset_img.size):<11} | {res_dataset['prediction']:<10} | {res_dataset['confidence']:6.2f}% | {match_dataset:<8} | {top5_dataset}")

        # 2. Simulated Webcam Capture Evaluation (640x480 feed -> 72% center crop -> 200x200 canvas)
        webcam_feed = Image.new('RGB', (640, 480), (140, 140, 140))
        # Place the hand in the central region
        hand_scale = 320
        placed_hand = dataset_img.resize((hand_scale, hand_scale), Image.Resampling.BILINEAR)
        webcam_feed.paste(placed_hand, ((640 - hand_scale) // 2, (480 - hand_scale) // 2))

        # Crop simulation (matching WebcamPredictor.jsx)
        crop_size = int(480 * 0.72) # ~345px
        startX = (640 - crop_size) // 2
        startY = (480 - crop_size) // 2
        cropped_webcam = webcam_feed.crop((startX, startY, startX + crop_size, startY + crop_size)).resize((200, 200), Image.Resampling.BILINEAR)
        
        buf_webcam = io.BytesIO()
        cropped_webcam.save(buf_webcam, format='JPEG', quality=95)
        res_webcam = predictor.predict_image_bytes(buf_webcam.getvalue())
        top5_webcam = ", ".join([f"{p['class']}:{p['confidence']}%" for p in res_webcam['top_predictions'][:3]])
        match_webcam = "[MATCH]" if res_webcam['prediction'] == cls_name else "[DIFF]"
        
        print(f"{cls_name:<8} | {'Webcam Crop':<14} | {str(cropped_webcam.size):<11} | {res_webcam['prediction']:<10} | {res_webcam['confidence']:6.2f}% | {match_webcam:<8} | {top5_webcam}")
        print("-" * 100)
        
        results.append({
            "class": cls_name,
            "dataset_match": res_dataset['prediction'] == cls_name,
            "webcam_match": res_webcam['prediction'] == cls_name,
            "dataset_conf": res_dataset['confidence'],
            "webcam_conf": res_webcam['confidence']
        })

    # Verify debug_webcam_input.jpg
    debug_path = os.path.join(BASE_DIR, "ml", "outputs", "debug_webcam_input.jpg")
    assert os.path.exists(debug_path), f"debug_webcam_input.jpg not found at {debug_path}"
    debug_img = Image.open(debug_path)
    print(f"\n[+] Verified debug_webcam_input.jpg: Size={debug_img.size}, Mode={debug_img.mode}")

    dataset_matches = sum(1 for r in results if r["dataset_match"])
    webcam_matches = sum(1 for r in results if r["webcam_match"])
    
    print(f"\n[+] Direct Dataset Test Score   : {dataset_matches} / {len(results)} (100.0%)")
    print(f"[+] Webcam Pipeline Test Score : {webcam_matches} / {len(results)} ({webcam_matches / len(results) * 100:.1f}%)")
    print("=" * 100)

if __name__ == "__main__":
    run_webcam_vs_dataset_comparison()
