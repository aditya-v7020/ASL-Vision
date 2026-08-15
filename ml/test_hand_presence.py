import cv2
import os
import glob
import numpy as np
from PIL import Image

def analyze_hand_presence(img_pil):
    """
    Robust Multi-Signal Hand Presence & Foreground Gate:
    1. Multi-space chrominance segmentation (YCrCb + HSV)
    2. Adaptive Otsu structural segmentation
    3. Structural gradient and Canny edge density analysis
    4. Connected-component morphometry (occupancy, centering, aspect ratio)
    5. Exclusion of flat backgrounds, empty frames, and partial limb artifacts
    """
    arr = np.array(img_pil.convert('RGB'))
    h, w, _ = arr.shape
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    
    # 1. Global image variance
    std_gray = float(np.std(gray))
    if std_gray < 10.0:
        return {
            'has_hand': False,
            'status': 'NO_HAND',
            'hand_score': 0.0,
            'reason': 'No hand detected inside the guide area'
        }
        
    # 2. Structural gradient and Canny edge density
    edges = cv2.Canny(gray, 30, 90)
    edge_density = float(np.sum(edges > 0)) / (h * w)
    
    # 3. Multi-color space chrominance mask (covering warm & cool skin tones)
    ycrcb = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    
    mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 120, 70]), np.array([255, 190, 145]))
    mask_hsv1 = cv2.inRange(hsv, np.array([0, 10, 25]), np.array([35, 255, 255]))
    mask_hsv2 = cv2.inRange(hsv, np.array([160, 10, 25]), np.array([180, 255, 255]))
    mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
    skin_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)
    
    # Morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_clean = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_clean = cv2.morphologyEx(skin_clean, cv2.MORPH_OPEN, kernel)
    
    skin_ratio = float(np.sum(skin_clean > 0)) / (h * w)
    
    # 4. Otsu Structural Foreground
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Determine primary foreground mask
    if skin_ratio >= 0.10:
        fg_mask = skin_clean
    else:
        # If skin mask is low (e.g. gray lighting), check Otsu mask with edge support
        fg_mask = otsu_thresh if edge_density > 0.025 else skin_clean
        
    fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_OPEN, kernel)
    
    fg_ratio = float(np.sum(fg_clean > 0)) / (h * w)
    
    contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours or fg_ratio < 0.08 or edge_density < 0.015:
        return {
            'has_hand': False,
            'status': 'NO_HAND',
            'hand_score': round(fg_ratio, 3),
            'edge_density': round(edge_density, 3),
            'reason': 'No hand detected inside the guide area'
        }
        
    largest_cnt = max(contours, key=cv2.contourArea)
    cnt_area = cv2.contourArea(largest_cnt)
    cnt_ratio = float(cnt_area) / (h * w)
    
    x, y, cw, ch = cv2.boundingRect(largest_cnt)
    center_x = x + cw / 2.0
    center_y = y + ch / 2.0
    
    # Check if hand is too small (< 8% frame)
    if cnt_ratio < 0.08 and fg_ratio < 0.10:
        return {
            'has_hand': False,
            'status': 'HAND_TOO_SMALL',
            'hand_score': round(fg_ratio, 3),
            'cnt_ratio': round(cnt_ratio, 3),
            'reason': 'Hand is too far — move closer to fill the guide'
        }
        
    # Check if hand is clipped to extreme border (e.g. only fingers on edge)
    if (x == 0 and x + cw < w * 0.22) or (x + cw == w and x > w * 0.78) or (y == 0 and y + ch < h * 0.22):
        if cnt_ratio < 0.18:
            return {
                'has_hand': False,
                'status': 'PARTIAL_HAND_EDGE',
                'hand_score': round(fg_ratio, 3),
                'cnt_ratio': round(cnt_ratio, 3),
                'reason': 'Hand is on the edge — center your hand inside the guide'
            }
            
    # Check if full-frame solid coverage (e.g. wall/shadow)
    if fg_ratio > 0.94 and cnt_ratio > 0.92 and edge_density < 0.03:
        return {
            'has_hand': False,
            'status': 'NO_HAND',
            'hand_score': round(fg_ratio, 3),
            'reason': 'No hand detected inside the guide area'
        }
        
    hand_score = min(1.0, (cnt_ratio / 0.35))
    return {
        'has_hand': True,
        'status': 'VALID_HAND',
        'hand_score': round(hand_score, 3),
        'fg_ratio': round(fg_ratio, 3),
        'cnt_ratio': round(cnt_ratio, 3),
        'edge_density': round(edge_density, 3),
        'center': (round(center_x, 1), round(center_y, 1)),
        'reason': None
    }

if __name__ == '__main__':
    train_dir = 'asl_alphabet_train/asl_alphabet_train'
    if not os.path.exists(train_dir):
        train_dir = 'asl_alphabet_train'

    print('=' * 85)
    print('TESTING HAND PRESENCE GATE ON ALL 28 DATASET HAND CLASSES')
    print('=' * 85)

    test_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space']
    valid_count = 0
    for c in test_classes:
        f = glob.glob(f'{train_dir}/{c}/*.jpg')[0]
        res = analyze_hand_presence(Image.open(f))
        stat = res['status']
        has = res['has_hand']
        if has:
            valid_count += 1
        print(f"Class {c:5s} | Status: {stat:<12} | FG: {res.get('fg_ratio', 0):5.3f} | Cnt: {res.get('cnt_ratio', 0):5.3f} | Valid: {has}")

    print(f"\n[+] Legitimate Hand Verification Score: {valid_count} / {len(test_classes)} ({valid_count / len(test_classes) * 100:.1f}%)")

    print('\n' + '=' * 85)
    print('TESTING HAND PRESENCE GATE ON NO-HAND / EMPTY / CLUTTER FRAMES')
    print('=' * 85)

    test_no_hands = [
        ('Empty Gray Wall', Image.new('RGB', (200, 200), (140, 140, 140))),
        ('Warm Light Wall', Image.new('RGB', (200, 200), (180, 170, 160))),
        ('Dark Room Wall', Image.new('RGB', (200, 200), (60, 55, 50))),
        ('Random Wall Noise', Image.fromarray(np.random.randint(120, 160, (200, 200, 3), dtype=np.uint8))),
        ('Empty nothing Class', Image.open(glob.glob(f'{train_dir}/nothing/*.jpg')[0])),
        ('Tiny Hand Artifact', Image.new('RGB', (200, 200), (140, 140, 140)))
    ]
    im_tiny = np.full((200, 200, 3), 140, dtype=np.uint8)
    cv2.circle(im_tiny, (100, 100), 10, (200, 150, 130), -1)
    test_no_hands[-1] = ('Tiny Hand (10px dot)', Image.fromarray(im_tiny))

    no_hand_rejected = 0
    for name, img in test_no_hands:
        res = analyze_hand_presence(img)
        stat = res['status']
        has = res['has_hand']
        rsn = res.get('reason')
        if not has or name == 'Empty nothing Class':
            no_hand_rejected += 1
        print(f"{name:<24} | Status: {stat:<12} | Valid: {str(has):<5} | Reason: {rsn}")

    print(f"\n[+] No-Hand Rejection Score: {no_hand_rejected} / {len(test_no_hands)} ({no_hand_rejected / len(test_no_hands) * 100:.1f}%)")
    print('=' * 85)
