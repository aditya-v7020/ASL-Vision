import os
import sys
import io
import glob
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.predictor import predictor
from ml.config import TRAIN_DIR, CLASSES_PATH, CLASSES

def run_automated_hand_presence_suite():
    print("=" * 105)
    print("AUTOMATED COMPREHENSIVE HAND-PRESENCE & FOREGROUND GATE TEST SUITE")
    print("=" * 105)

    predictor.load_model()
    assert predictor.model is not None, "Predictor model failed to load!"

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    # -------------------------------------------------------------
    # TEST SUITE 1: NO-HAND / EMPTY / CLUTTER FRAMES (CRITICAL TEST)
    # -------------------------------------------------------------
    print("\n[TEST SUITE 1] EVALUATING NO-HAND & CLUTTER FRAMES (MUST NEVER PREDICT ASL LETTERS)")
    print("-" * 105)
    print(f"{'Condition':<32} | {'Predicted':<12} | {'Conf':<7} | {'Uncertain':<10} | {'Status':<16} | {'Reason'}")
    print("-" * 105)

    no_hand_cases = []
    # 1. Plain Gray Wall
    no_hand_cases.append(('Empty Gray Wall', Image.new('RGB', (200, 200), (140, 140, 140))))
    # 2. Warm Daylight Wall
    no_hand_cases.append(('Warm Daylight Wall', Image.new('RGB', (200, 200), (185, 175, 165))))
    # 3. Dim Room Wall
    no_hand_cases.append(('Dim Ambient Room Wall', Image.new('RGB', (200, 200), (55, 52, 48))))
    # 4. Textured Wallpaper Noise
    no_hand_cases.append(('Wallpaper Texture Noise', Image.fromarray(np.random.randint(120, 155, (200, 200, 3), dtype=np.uint8))))
    # 5. Torso / Shirt in Frame (no hand)
    torso_img = Image.new('RGB', (200, 200), (140, 140, 140))
    d_torso = ImageDraw.Draw(torso_img)
    d_torso.rectangle([0, 90, 200, 200], fill=(45, 65, 110))
    no_hand_cases.append(('Torso / Shirt in Frame', torso_img))
    # 6. Furniture Edge / Room Corner
    furn_img = Image.new('RGB', (200, 200), (150, 145, 140))
    d_furn = ImageDraw.Draw(furn_img)
    d_furn.rectangle([0, 120, 90, 200], fill=(75, 45, 25))
    d_furn.line([(0, 120), (200, 135)], fill=(50, 50, 50), width=3)
    no_hand_cases.append(('Room / Furniture Edge', furn_img))
    # 7. Tiny 8px Skin Dot (Sensor Noise)
    tiny_dot = Image.new('RGB', (200, 200), (140, 140, 140))
    d_dot = ImageDraw.Draw(tiny_dot)
    d_dot.ellipse([96, 96, 104, 104], fill=(200, 150, 130))
    no_hand_cases.append(('Tiny Noise Artifact (8px)', tiny_dot))
    # 8. Dataset "nothing" sample
    nothing_f = glob.glob(os.path.join(TRAIN_DIR, 'nothing', '*.jpg'))[0]
    no_hand_cases.append(('Dataset "nothing" Sample', Image.open(nothing_f)))

    suite1_passes = 0
    for name, img in no_hand_cases:
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=95)
        res = predictor.predict_image_bytes(buf.getvalue())
        
        pred = res['prediction']
        conf = res['confidence']
        is_unc = res['is_uncertain']
        hp = res.get('hand_presence', {})
        rsn = res.get('uncertainty_reason') or 'None'
        
        # Success criteria for Suite 1: Must NOT predict an ASL letter (must be "nothing" or "Uncertain...")
        is_safe = (pred == 'nothing' or pred.startswith('Uncertain'))
        if is_safe:
            suite1_passes += 1
            status_tag = '[PASSED - GATED]'
        else:
            status_tag = '[FAILED - LEAKED]'
            
        print(f"{name:<32} | {pred:<12} | {conf:5.1f}% | {str(is_unc):<10} | {status_tag:<16} | {rsn[:38]}")

    print(f"\n[+] Suite 1 (No-Hand Protection) Score: {suite1_passes} / {len(no_hand_cases)} ({suite1_passes / len(no_hand_cases) * 100:.1f}%)")

    # -------------------------------------------------------------
    # TEST SUITE 2: PARTIAL HAND & INCORRECT POSITIONING FRAMES
    # -------------------------------------------------------------
    print("\n[TEST SUITE 2] EVALUATING PARTIAL HANDS & INCORRECT POSITIONING")
    print("-" * 105)
    print(f"{'Condition':<32} | {'Predicted':<12} | {'Conf':<7} | {'Uncertain':<10} | {'Status':<16} | {'Reason'}")
    print("-" * 105)

    partial_cases = []
    # Sample real hand from dataset
    o_hand = Image.open(glob.glob(os.path.join(TRAIN_DIR, 'O', '*.jpg'))[0]).convert('RGB')
    
    # 1. Distant tiny hand (taking only 5% of crop box)
    distant_frame = Image.new('RGB', (200, 200), (140, 140, 140))
    distant_frame.paste(o_hand.resize((45, 45)), (78, 78))
    partial_cases.append(('Distant Tiny Hand (~5% Area)', distant_frame))
    
    # 2. Hand clipped at extreme right border (only 15% visible)
    right_clip = Image.new('RGB', (200, 200), (140, 140, 140))
    right_clip.paste(o_hand.resize((150, 150)), (170, 25))
    partial_cases.append(('Hand Clipped Right Edge', right_clip))
    
    # 3. Hand clipped at extreme top border
    top_clip = Image.new('RGB', (200, 200), (140, 140, 140))
    top_clip.paste(o_hand.resize((150, 150)), (25, -110))
    partial_cases.append(('Hand Clipped Top Edge', top_clip))

    # 4. Severe Motion Blur on Hand
    blur_hand = o_hand.filter(ImageFilter.GaussianBlur(radius=8))
    partial_cases.append(('Severe Motion Blur (r=8)', blur_hand))

    suite2_passes = 0
    for name, img in partial_cases:
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=95)
        res = predictor.predict_image_bytes(buf.getvalue())
        
        pred = res['prediction']
        conf = res['confidence']
        is_unc = res['is_uncertain']
        rsn = res.get('uncertainty_reason') or 'None'
        
        # Success criteria for Suite 2: Must flag uncertainty and give guidance
        is_safe = (is_unc is True)
        if is_safe:
            suite2_passes += 1
            status_tag = '[PASSED - GATED]'
        else:
            status_tag = '[FAILED - UNGATED]'
            
        print(f"{name:<32} | {pred:<12} | {conf:5.1f}% | {str(is_unc):<10} | {status_tag:<16} | {rsn[:38]}")

    print(f"\n[+] Suite 2 (Degraded/Partial Protection) Score: {suite2_passes} / {len(partial_cases)} ({suite2_passes / len(partial_cases) * 100:.1f}%)")

    # -------------------------------------------------------------
    # TEST SUITE 3: LEGITIMATE ASL HAND SIGNS (MUST CLASSIFY CORRECTLY)
    # -------------------------------------------------------------
    print("\n[TEST SUITE 3] EVALUATING LEGITIMATE ASL SIGNS ACROSS KEY CLASSES")
    print("-" * 105)
    print(f"{'Class':<8} | {'Predicted':<12} | {'Conf':<7} | {'Proto Match':<12} | {'Proto Sim':<10} | {'Status':<16}")
    print("-" * 105)

    key_asl_classes = ['A', 'O', 'G', 'H', 'B', 'K', 'I', 'D', 'F', 'U', 'V', 'M', 'N', 'T', 'Y', 'del', 'space', 'nothing']
    suite3_passes = 0

    for cls in key_asl_classes:
        fpath = glob.glob(os.path.join(TRAIN_DIR, cls, '*.jpg'))[0]
        img = Image.open(fpath).convert('RGB')
        
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=95)
        res = predictor.predict_image_bytes(buf.getvalue())
        
        pred = res['prediction']
        conf = res['confidence']
        proto_cls = res.get('prototype_match') or 'None'
        proto_sim = res.get('prototype_similarity') or 0.0
        
        is_match = (pred == cls)
        if is_match:
            suite3_passes += 1
            status_tag = '[MATCH]'
        else:
            status_tag = '[DIFF]'
            
        print(f"{cls:<8} | {pred:<12} | {conf:5.1f}% | {proto_cls:<12} | {proto_sim:<10.4f} | {status_tag:<16}")

    print(f"\n[+] Suite 3 (Legitimate ASL Classification) Score: {suite3_passes} / {len(key_asl_classes)} ({suite3_passes / len(key_asl_classes) * 100:.1f}%)")
    print("=" * 105)

if __name__ == '__main__':
    run_automated_hand_presence_suite()
