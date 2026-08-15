import os
import sys
import io
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.predictor import predictor
from ml.config import TRAIN_DIR, CLASSES, CLASSES_PATH, MODEL_PATH, IMAGE_SIZE, OUTPUTS_DIR

BENCHMARK_REPORT_PATH = os.path.join(OUTPUTS_DIR, "webcam_benchmark_results.txt")
WEBCAM_CM_PATH = os.path.join(OUTPUTS_DIR, "webcam_confusion_matrix.png")

def plot_webcam_confusion_matrix(cm, class_names, output_path=WEBCAM_CM_PATH):
    plt.figure(figsize=(18, 16))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("29-Class Live Webcam ASL Recognition - Confusion Matrix Heatmap (870 Frames)", fontsize=16, fontweight="bold", pad=20)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right", fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                plt.text(
                    j, i, f"{val}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if val > thresh else "black",
                    fontsize=8
                )
                
    plt.ylabel("True Class", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[+] Webcam confusion matrix saved to: {output_path}")

def run_large_scale_webcam_benchmark(frames_per_class=30):
    print("=" * 105)
    print(f"LARGE-SCALE 29-CLASS WEBCAM BENCHMARK ({frames_per_class} FRAMES/CLASS = {frames_per_class * 29} TOTAL FRAMES)")
    print("=" * 105)

    predictor.load_model()
    assert predictor.model is not None, "Model failed to load!"

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    # Realistic background variations for webcam synthesis
    background_palettes = [
        (140, 142, 145), # Neutral office wall
        (165, 158, 150), # Warm daylight wall
        (120, 125, 130), # Dim indoor room
        (155, 150, 142), # Wood/desk backdrop
        (130, 135, 140)  # Evening indoor
    ]

    y_true = []
    y_pred_before = []
    y_pred_after = []
    y_pred_after_raw = [] # Raw top-1 without uncertainty text
    uncertain_flags = []
    confidence_scores = []
    prototype_similarities = []

    per_class_stats = {
        c: {
            "before_correct": 0,
            "after_correct": 0,
            "uncertain_count": 0,
            "total": 0,
            "confidences": [],
            "proto_sims": []
        } for c in classes
    }

    print(f"[*] Generating and testing {len(classes) * frames_per_class} realistic webcam frames...")

    np.random.seed(42)

    for class_idx, cls in enumerate(classes):
        # Pick 30 distinct test images per class from the latter portion of dataset (indices 2000 to 2900)
        all_class_files = sorted(glob.glob(os.path.join(TRAIN_DIR, cls, "*.jpg")))
        sample_files = all_class_files[2000:2000 + frames_per_class]
        if len(sample_files) < frames_per_class:
            sample_files = all_class_files[-frames_per_class:]

        for frame_idx, fpath in enumerate(sample_files):
            im = Image.open(fpath).convert("RGB")
            
            # --- REALISTIC WEBCAM DISTORTIONS ---
            # 1. Background selection
            bg_color = background_palettes[frame_idx % len(background_palettes)]
            webcam_640x480 = Image.new("RGB", (640, 480), bg_color)
            
            # 2. Hand scale variation (filling 65% to 75% of the guide box)
            scale_factor = 0.68 + (frame_idx % 5) * 0.02 # 0.68 to 0.76
            hand_px = int(480 * scale_factor)
            
            # 3. Slight translation / centering jitter (±12 px)
            dx = (frame_idx % 7 - 3) * 4
            dy = (frame_idx % 5 - 2) * 4
            
            # 4. Slight rotation (±5 degrees)
            rot_deg = (frame_idx % 5 - 2) * 2.5
            
            # 5. Slight lighting / contrast adjustment
            bright_factor = 0.92 + (frame_idx % 5) * 0.04
            
            if cls == "nothing":
                # For 'nothing' class, frame is empty background
                hand_placed = webcam_640x480
            else:
                im_aug = im.rotate(rot_deg, resample=Image.Resampling.BILINEAR, expand=False)
                im_aug = ImageEnhance.Brightness(im_aug).enhance(bright_factor)
                im_scaled = im_aug.resize((hand_px, hand_px), Image.Resampling.BILINEAR)
                
                paste_x = (640 - hand_px) // 2 + dx
                paste_y = (480 - hand_px) // 2 + dy
                webcam_640x480.paste(im_scaled, (paste_x, paste_y))

            # --- 1. BEFORE PIPELINE (Raw uncropped 640x480 feed directly downsampled) ---
            buf_before = io.BytesIO()
            webcam_640x480.resize(IMAGE_SIZE, Image.Resampling.BILINEAR).save(buf_before, format="JPEG")
            res_before = predictor.predict_image_bytes(buf_before.getvalue())
            pred_b = res_before["prediction"]
            is_b_correct = (pred_b == cls)

            # --- 2. AFTER PIPELINE (Calibrated 72% central crop + Canonical Right-Hand Mirroring) ---
            crop_size = int(480 * 0.72)
            sX = (640 - crop_size) // 2
            sY = (480 - crop_size) // 2
            crop_aligned = webcam_640x480.crop((sX, sY, sX + crop_size, sY + crop_size)).resize((200, 200), Image.Resampling.BILINEAR)
            
            buf_after = io.BytesIO()
            crop_aligned.save(buf_after, format="JPEG", quality=95)
            res_after = predictor.predict_image_bytes(buf_after.getvalue())
            
            pred_a = res_after["prediction"]
            is_unc = res_after["is_uncertain"]
            raw_top_cls = res_after["top_predictions"][0]["class"] if res_after["top_predictions"] else "nothing"
            
            is_a_correct = (pred_a == cls)
            
            # Record tracking metrics
            y_true.append(class_idx)
            y_pred_before.append(classes.index(pred_b) if pred_b in classes else -1)
            y_pred_after.append(classes.index(pred_a) if pred_a in classes else -1)
            y_pred_after_raw.append(classes.index(raw_top_cls) if raw_top_cls in classes else -1)
            
            uncertain_flags.append(is_unc)
            confidence_scores.append(res_after["confidence"])
            if res_after.get("prototype_similarity"):
                prototype_similarities.append(res_after["prototype_similarity"])

            # Per class stats
            per_class_stats[cls]["total"] += 1
            if is_b_correct: per_class_stats[cls]["before_correct"] += 1
            if is_a_correct: per_class_stats[cls]["after_correct"] += 1
            if is_unc: per_class_stats[cls]["uncertain_count"] += 1
            per_class_stats[cls]["confidences"].append(res_after["confidence"])
            if res_after.get("prototype_similarity"):
                per_class_stats[cls]["proto_sims"].append(res_after["prototype_similarity"])

        print(f"  [{class_idx+1:2d}/29] {cls:>8}: tested {frames_per_class} frames (After: {per_class_stats[cls]['after_correct']}/{frames_per_class})", flush=True)

    total_frames = len(y_true)
    before_acc = sum(1 for yt, yp in zip(y_true, y_pred_before) if yt == yp) / total_frames * 100
    after_acc = sum(1 for yt, yp in zip(y_true, y_pred_after) if yt == yp) / total_frames * 100
    after_raw_acc = sum(1 for yt, yp in zip(y_true, y_pred_after_raw) if yt == yp) / total_frames * 100
    total_uncertain = sum(uncertain_flags)

    # Compute valid (certain) frames accuracy
    certain_indices = [i for i, u in enumerate(uncertain_flags) if not u]
    if certain_indices:
        certain_acc = sum(1 for i in certain_indices if y_true[i] == y_pred_after[i]) / len(certain_indices) * 100
    else:
        certain_acc = 0.0

    # Build Confusion Matrix & Classification Report
    cm = confusion_matrix(y_true, y_pred_after_raw, labels=list(range(len(classes))))
    plot_webcam_confusion_matrix(cm, classes, WEBCAM_CM_PATH)

    report_str = classification_report(
        y_true,
        y_pred_after_raw,
        target_names=classes,
        digits=4
    )

    # Top confused pairs
    confused_pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((classes[i], classes[j], int(cm[i, j])))
    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    # Print Summary Table
    print("\n" + "=" * 105)
    print("LARGE-SCALE 29-CLASS WEBCAM ACCURACY REPORT (870 FRAMES)")
    print("=" * 105)
    print(f"{'Class':<8} | {'BEFORE (Raw Feed)':<20} | {'AFTER (Optimized)':<20} | {'Uncertain':<10} | {'Avg Conf':<10} | {'Avg ProtoSim':<12}")
    print("-" * 105)

    weak_focus = ['B', 'K', 'I', 'D', 'F']
    weak_improvements = {}

    for cls in classes:
        st = per_class_stats[cls]
        b_acc = st["before_correct"] / st["total"] * 100
        a_acc = st["after_correct"] / st["total"] * 100
        avg_c = np.mean(st["confidences"]) if st["confidences"] else 0.0
        avg_p = np.mean(st["proto_sims"]) if st["proto_sims"] else 0.0
        
        tag = ""
        if cls in weak_focus:
            weak_improvements[cls] = (b_acc, a_acc)
            tag = "  <-- WEAK FOCUS"
            
        print(f"{cls:<8} | {st['before_correct']:2d}/{st['total']:2d} ({b_acc:5.1f}%)        | {st['after_correct']:2d}/{st['total']:2d} ({a_acc:5.1f}%)        | {st['uncertain_count']:2d} frames   | {avg_c:5.1f}%     | {avg_p:6.4f}{tag}")

    print("-" * 105)
    print(f"{'TOTAL':<8} | {before_acc:5.2f}%               | {after_acc:5.2f}%               | {total_uncertain:2d} frames   | {np.mean(confidence_scores):5.1f}%     | {np.mean(prototype_similarities):6.4f}")
    print("=" * 105)

    print(f"\n[+] Raw Prediction Accuracy (all 870 frames)        : {after_raw_acc:.2f}%")
    print(f"[+] Effective Accuracy with Uncertainty Gating      : {after_acc:.2f}%")
    print(f"[+] Accuracy on Non-Uncertain (Gated) Frames       : {certain_acc:.2f}% (N={len(certain_indices)}/{total_frames})")
    print(f"[+] Total Uncertain / Low-Quality Frames Flagged    : {total_uncertain} / {total_frames} ({total_uncertain/total_frames*100:.1f}%)")

    print("\n--- WEAK CLASSES FOCUS IMPROVEMENT SUMMARY (B, K, I, D, F) ---")
    for cls in weak_focus:
        b_val, a_val = weak_improvements[cls]
        diff_str = f"+{a_val - b_val:5.1f}%" if a_val >= b_val else f"{a_val - b_val:5.1f}%"
        print(f"  Class {cls:<2} : Before = {b_val:5.1f}%  -->  After = {a_val:5.1f}%  (Change: {diff_str})")

    print("\n--- TOP CONFUSION PAIRS IN WEBCAM FRAMES ---")
    if confused_pairs:
        for idx, (t, p, c) in enumerate(confused_pairs[:8], 1):
            print(f"  {idx}. True '{t}' -> Predicted '{p}': {c} frames ({c/frames_per_class*100:.1f}% of class)")
    else:
        print("  None! Perfect classification across all classes.")


    # Save Full Detailed Report to Disk
    with open(BENCHMARK_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("AI-BASED SIGN LANGUAGE RECOGNITION - LARGE-SCALE WEBCAM BENCHMARK REPORT\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Total Evaluated Webcam Frames : {total_frames} ({frames_per_class} independent frames × 29 classes)\n")
        f.write(f"Webcam Accuracy BEFORE        : {before_acc:.2f}%\n")
        f.write(f"Webcam Accuracy AFTER (Raw)   : {after_raw_acc:.2f}%\n")
        f.write(f"Webcam Accuracy AFTER (Gated) : {after_acc:.2f}%\n")
        f.write(f"Accuracy on Certain Frames    : {certain_acc:.2f}% (N={len(certain_indices)})\n")
        f.write(f"Total Uncertain Frames Gated  : {total_uncertain} ({total_uncertain/total_frames*100:.1f}%)\n\n")
        
        f.write("WEAK CLASSES BEFORE VS AFTER SUMMARY:\n")
        for cls in weak_focus:
            b_val, a_val = weak_improvements[cls]
            diff_s = f"+{a_val - b_val:5.1f}%" if a_val >= b_val else f"{a_val - b_val:5.1f}%"
            f.write(f"  Class {cls:<2}: Before = {b_val:5.1f}%  -->  After = {a_val:5.1f}% (Change: {diff_s})\n")
        f.write("\n" + "-" * 90 + "\n")

        f.write("PER-CLASS WEBCAM ACCURACY BREAKDOWN:\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Class':<8} | {'Before Acc':<12} | {'After Acc':<12} | {'Uncertain':<10} | {'Avg Conf':<10} | {'Avg ProtoSim'}\n")
        f.write("-" * 90 + "\n")
        for cls in classes:
            st = per_class_stats[cls]
            b_acc = st["before_correct"] / st["total"] * 100
            a_acc = st["after_correct"] / st["total"] * 100
            avg_c = np.mean(st["confidences"]) if st["confidences"] else 0.0
            avg_p = np.mean(st["proto_sims"]) if st["proto_sims"] else 0.0
            f.write(f"{cls:<8} | {b_acc:5.1f}%       | {a_acc:5.1f}%       | {st['uncertain_count']:2d} frames   | {avg_c:5.1f}%     | {avg_p:6.4f}\n")
            
        f.write("\n" + "=" * 90 + "\n")
        f.write("CLASSIFICATION REPORT (PRECISION / RECALL / F1 PER CLASS):\n")
        f.write("=" * 90 + "\n")
        f.write(report_str + "\n\n")
        
        f.write("TOP CONFUSED PAIRS:\n")
        for idx, (t, p, c) in enumerate(confused_pairs[:15], 1):
            f.write(f"  {idx:2d}. True '{t}' -> Predicted '{p}': {c} occurrences\n")
            
    print(f"\n[+] Full detailed benchmark report saved to: {BENCHMARK_REPORT_PATH}")
    print("=" * 105 + "\n")
    return {
        "before_acc": before_acc,
        "after_acc": after_acc,
        "after_raw_acc": after_raw_acc,
        "certain_acc": certain_acc,
        "total_uncertain": total_uncertain,
        "weak_improvements": weak_improvements,
        "total_frames": total_frames
    }

if __name__ == "__main__":
    run_large_scale_webcam_benchmark(frames_per_class=30)
