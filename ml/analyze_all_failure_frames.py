import os
import sys
import io
import json
import glob
import numpy as np
from PIL import Image, ImageEnhance

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.predictor import predictor
from ml.config import TRAIN_DIR, CLASSES_PATH, OUTPUTS_DIR, IMAGE_SIZE

def analyze_all_failures():
    print("=" * 105)
    print("DEEP DIAGNOSTIC FAILURE ANALYSIS ACROSS ALL 870 WEBCAM BENCHMARK FRAMES")
    print("=" * 105)

    predictor.load_model()
    assert predictor.model is not None, "Model failed to load!"

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    background_palettes = [
        (140, 142, 145), # Neutral office wall
        (165, 158, 150), # Warm daylight wall
        (120, 125, 130), # Dim indoor room
        (155, 150, 142), # Wood/desk backdrop
        (130, 135, 140)  # Evening indoor
    ]

    frames_per_class = 30
    failed_records = []
    uncertain_records = []
    correct_records = []

    confusion_groups = {}
    per_class_summary = {c: {"total": 0, "correct": 0, "incorrect": 0, "uncertain": 0} for c in classes}

    print(f"[*] Analyzing {len(classes) * frames_per_class} webcam frames...")

    np.random.seed(42)

    for class_idx, cls in enumerate(classes):
        all_class_files = sorted(glob.glob(os.path.join(TRAIN_DIR, cls, "*.jpg")))
        sample_files = all_class_files[2000:2000 + frames_per_class]
        if len(sample_files) < frames_per_class:
            sample_files = all_class_files[-frames_per_class:]

        for frame_idx, fpath in enumerate(sample_files):
            im = Image.open(fpath).convert("RGB")
            
            # Reconstruct exact benchmark synthesis
            bg_color = background_palettes[frame_idx % len(background_palettes)]
            webcam_640x480 = Image.new("RGB", (640, 480), bg_color)
            
            scale_factor = 0.68 + (frame_idx % 5) * 0.02
            hand_px = int(480 * scale_factor)
            
            dx = (frame_idx % 7 - 3) * 4
            dy = (frame_idx % 5 - 2) * 4
            rot_deg = (frame_idx % 5 - 2) * 2.5
            bright_factor = 0.92 + (frame_idx % 5) * 0.04
            
            if cls == "nothing":
                hand_placed = webcam_640x480
            else:
                im_aug = im.rotate(rot_deg, resample=Image.Resampling.BILINEAR, expand=False)
                im_aug = ImageEnhance.Brightness(im_aug).enhance(bright_factor)
                im_scaled = im_aug.resize((hand_px, hand_px), Image.Resampling.BILINEAR)
                
                paste_x = (640 - hand_px) // 2 + dx
                paste_y = (480 - hand_px) // 2 + dy
                webcam_640x480.paste(im_scaled, (paste_x, paste_y))

            # Calibrated Crop
            crop_size = int(480 * 0.72)
            sX = (640 - crop_size) // 2
            sY = (480 - crop_size) // 2
            crop_aligned = webcam_640x480.crop((sX, sY, sX + crop_size, sY + crop_size)).resize((200, 200), Image.Resampling.BILINEAR)
            
            buf = io.BytesIO()
            crop_aligned.save(buf, format="JPEG", quality=95)
            res = predictor.predict_image_bytes(buf.getvalue())
            
            pred = res["prediction"]
            conf = res["confidence"]
            is_unc = res["is_uncertain"]
            raw_top = res["top_predictions"][0]["class"] if res["top_predictions"] else "nothing"
            top_preds = res["top_predictions"]
            top2_cls = top_preds[1]["class"] if len(top_preds) > 1 else None
            top2_conf = top_preds[1]["confidence"] if len(top_preds) > 1 else 0.0
            margin = round(conf - top2_conf, 2)
            
            proto_cls = res.get("prototype_match")
            proto_sim = res.get("prototype_similarity")
            hp = res.get("hand_presence", {})
            qm = res.get("quality_metrics", {})
            rsn = res.get("uncertainty_reason")

            record = {
                "frame_id": f"{cls}_{frame_idx:02d}",
                "true_class": cls,
                "file_source": os.path.basename(fpath),
                "predicted_class": pred,
                "raw_top_class": raw_top,
                "confidence": conf,
                "top2_class": top2_cls,
                "top2_confidence": top2_conf,
                "margin": margin,
                "is_uncertain": is_unc,
                "uncertainty_reason": rsn,
                "prototype_match": proto_cls,
                "prototype_similarity": proto_sim,
                "hand_presence_status": hp.get("status"),
                "hand_score": hp.get("hand_score"),
                "fg_ratio": hp.get("fg_ratio"),
                "cnt_ratio": hp.get("cnt_ratio"),
                "edge_density": hp.get("edge_density"),
                "quality_sharpness": qm.get("sharpness"),
                "quality_brightness": qm.get("brightness"),
                "quality_contrast": qm.get("contrast"),
                "scale_factor": round(scale_factor, 2),
                "dx": dx,
                "dy": dy,
                "rotation": rot_deg,
                "brightness_factor": round(bright_factor, 2)
            }

            per_class_summary[cls]["total"] += 1

            if cls == "nothing":
                # For 'nothing', pred == 'nothing' is correct
                if pred == "nothing":
                    per_class_summary[cls]["correct"] += 1
                    correct_records.append(record)
                else:
                    per_class_summary[cls]["incorrect"] += 1
                    failed_records.append(record)
            else:
                if is_unc:
                    per_class_summary[cls]["uncertain"] += 1
                    uncertain_records.append(record)
                    # Track confusion
                    pair_key = f"{cls} -> Uncertain ({rsn})"
                    confusion_groups[pair_key] = confusion_groups.get(pair_key, 0) + 1
                elif pred == cls:
                    per_class_summary[cls]["correct"] += 1
                    correct_records.append(record)
                else:
                    per_class_summary[cls]["incorrect"] += 1
                    failed_records.append(record)
                    pair_key = f"{cls} -> {pred}"
                    confusion_groups[pair_key] = confusion_groups.get(pair_key, 0) + 1

    # Save JSON report
    failure_report_data = {
        "total_frames": len(classes) * frames_per_class,
        "total_correct": len(correct_records),
        "total_incorrect": len(failed_records),
        "total_uncertain": len(uncertain_records),
        "raw_accuracy": round(len(correct_records) / (len(classes) * frames_per_class) * 100, 2),
        "certain_accuracy": round(len(correct_records) / max(1, len(correct_records) + len(failed_records)) * 100, 2),
        "per_class_summary": per_class_summary,
        "confusion_groups": dict(sorted(confusion_groups.items(), key=lambda x: x[1], reverse=True)),
        "failed_frames": failed_records,
        "uncertain_frames": uncertain_records
    }

    json_path = os.path.join(OUTPUTS_DIR, "webcam_failure_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(failure_report_data, f, indent=2)

    # Save TXT report
    txt_path = os.path.join(OUTPUTS_DIR, "webcam_failure_analysis.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("WEBCAM INFERENCE FAILURE & UNCERTAINTY ANALYSIS REPORT\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Total Evaluated Frames : {failure_report_data['total_frames']}\n")
        f.write(f"Total Correct Frames   : {failure_report_data['total_correct']} ({failure_report_data['raw_accuracy']}%)\n")
        f.write(f"Total Incorrect Frames : {failure_report_data['total_incorrect']}\n")
        f.write(f"Total Uncertain Frames : {failure_report_data['total_uncertain']}\n")
        f.write(f"Certain-Frame Accuracy : {failure_report_data['certain_accuracy']}%\n\n")
        
        f.write("-" * 90 + "\n")
        f.write("RANKED CONFUSION GROUPS & CAUSES:\n")
        f.write("-" * 90 + "\n")
        for pair, count in sorted(confusion_groups.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {pair:<48} : {count:2d} occurrences\n")
            
        f.write("\n" + "-" * 90 + "\n")
        f.write("DETAILED BREAKDOWN OF ALL INCORRECT & UNCERTAIN FRAMES:\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Frame ID':<10} | {'True':<6} | {'Pred':<12} | {'Conf':<7} | {'Top-2':<6} | {'Margin':<7} | {'Proto':<6} | {'Sim':<6} | {'Status/Reason'}\n")
        f.write("-" * 90 + "\n")
        for rec in failed_records + uncertain_records:
            f.write(f"{rec['frame_id']:<10} | {rec['true_class']:<6} | {rec['predicted_class']:<12} | {rec['confidence']:5.1f}% | {str(rec['top2_class']):<6} | {rec['margin']:5.1f}% | {str(rec['prototype_match']):<6} | {str(rec['prototype_similarity']):<6} | {str(rec['uncertainty_reason'])[:25]}\n")

    print(f"\n[+] Detailed failure analysis JSON saved to: {json_path}")
    print(f"[+] Detailed failure analysis TXT saved to: {txt_path}")
    print("\n--- TOP CONFUSION PAIRS ---")
    for pair, count in sorted(confusion_groups.items(), key=lambda x: x[1], reverse=True)[:12]:
        print(f"  {pair:<45} : {count:2d} frames")
    print("=" * 105)

if __name__ == "__main__":
    analyze_all_failures()
