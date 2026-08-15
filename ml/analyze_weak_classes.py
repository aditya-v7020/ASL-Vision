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
from ml.config import TRAIN_DIR, CLASSES_PATH, MODEL_PATH, IMAGE_SIZE

predictor.load_model()
with open(CLASSES_PATH, 'r') as f:
    classes = json.load(f)

weak_classes = ['B', 'K', 'I', 'D', 'F']

print("=== ANALYZING WEBCAM CROP PREDICTIONS FOR B, K, I, D, F ===")

for cls in weak_classes:
    files = sorted(glob.glob(os.path.join(TRAIN_DIR, cls, '*.jpg')))[2000:2020]
    matches = 0
    print(f"\n--- CLASS {cls} (20 frames) ---")
    for idx, f in enumerate(files):
        im = Image.open(f).convert('RGB')
        
        webcam_feed = Image.new('RGB', (640, 480), (135, 138, 142))
        hand_scaled = im.resize((330, 330), Image.Resampling.BILINEAR)
        webcam_feed.paste(hand_scaled, ((640 - 330) // 2, (480 - 330) // 2))
        
        crop_size = int(480 * 0.72)
        sX = (640 - crop_size) // 2
        sY = (480 - crop_size) // 2
        crop_std = webcam_feed.crop((sX, sY, sX + crop_size, sY + crop_size)).resize((200, 200), Image.Resampling.BILINEAR)
        
        buf = io.BytesIO()
        crop_std.save(buf, format='JPEG')
        res = predictor.predict_image_bytes(buf.getvalue())
        
        pred_cls = res['prediction']
        conf = res['confidence']
        is_match = (pred_cls == cls)
        if is_match:
            matches += 1
            
        top2_cls = res['top_predictions'][1]['class'] if len(res['top_predictions']) > 1 else 'None'
        top2_conf = res['top_predictions'][1]['confidence'] if len(res['top_predictions']) > 1 else 0.0
        proto_cls = res.get('prototype_match', 'None')
        proto_sim = res.get('prototype_similarity', 0.0)
        
        tag = "[MATCH]" if is_match else "[DIFF]"
        print(f"Sample {idx:2d}: True={cls} | Pred={pred_cls:<6} ({conf:5.1f}%) | Top-2={top2_cls:<6} ({top2_conf:5.1f}%) | Proto={proto_cls:<6} (sim={proto_sim:.4f}) | {tag}")
        
    print(f"Total {cls} Match: {matches}/20 ({matches/20*100:.1f}%)")
