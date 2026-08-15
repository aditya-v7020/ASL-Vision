"""
Targeted Failure Analysis for Weak Webcam ASL Classes
=====================================================
Focuses on K, A, F, M, B, R, E, I with:
- Full top-5 probability distribution per frame
- Full prototype ranking (all 29 classes, not just top-1)
- Visual gallery: correct vs incorrect frames saved as images
- Confusion matrix for weak classes
- Comparison of correct vs incorrect frame statistics
"""
import os
import sys
import io
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from collections import defaultdict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.predictor import predictor, preprocess_image
from ml.config import TRAIN_DIR, CLASSES_PATH, OUTPUTS_DIR, IMAGE_SIZE

WEAK_CLASSES = ["K", "A", "F", "M", "B", "R", "E", "I"]
GALLERY_DIR = os.path.join(OUTPUTS_DIR, "failure_galleries")
ANALYSIS_JSON = os.path.join(OUTPUTS_DIR, "targeted_class_failure_analysis.json")
ANALYSIS_TXT = os.path.join(OUTPUTS_DIR, "targeted_class_failure_analysis.txt")

# Same benchmark synthesis parameters as large_scale_webcam_benchmark.py
BACKGROUND_PALETTES = [
    (140, 142, 145),
    (165, 158, 150),
    (120, 125, 130),
    (155, 150, 142),
    (130, 135, 140)
]
FRAMES_PER_CLASS = 30


def synthesize_webcam_frame(im, cls, frame_idx):
    """Exact replica of the benchmark's webcam frame synthesis."""
    bg_color = BACKGROUND_PALETTES[frame_idx % len(BACKGROUND_PALETTES)]
    webcam_640x480 = Image.new("RGB", (640, 480), bg_color)

    scale_factor = 0.68 + (frame_idx % 5) * 0.02
    hand_px = int(480 * scale_factor)
    dx = (frame_idx % 7 - 3) * 4
    dy = (frame_idx % 5 - 2) * 4
    rot_deg = (frame_idx % 5 - 2) * 2.5
    bright_factor = 0.92 + (frame_idx % 5) * 0.04

    if cls == "nothing":
        pass
    else:
        im_aug = im.rotate(rot_deg, resample=Image.Resampling.BILINEAR, expand=False)
        im_aug = ImageEnhance.Brightness(im_aug).enhance(bright_factor)
        im_scaled = im_aug.resize((hand_px, hand_px), Image.Resampling.BILINEAR)
        paste_x = (640 - hand_px) // 2 + dx
        paste_y = (480 - hand_px) // 2 + dy
        webcam_640x480.paste(im_scaled, (paste_x, paste_y))

    # Calibrated crop (same as benchmark)
    crop_size = int(480 * 0.72)
    sX = (640 - crop_size) // 2
    sY = (480 - crop_size) // 2
    crop_aligned = webcam_640x480.crop((sX, sY, sX + crop_size, sY + crop_size)).resize(
        (200, 200), Image.Resampling.BILINEAR
    )

    aug_params = {
        "scale_factor": round(scale_factor, 2),
        "hand_px": hand_px,
        "dx": dx, "dy": dy,
        "rotation": rot_deg,
        "brightness_factor": round(bright_factor, 2),
        "crop_size": crop_size,
        "crop_x": sX, "crop_y": sY,
        "paste_x": (640 - hand_px) // 2 + dx,
        "paste_y": (480 - hand_px) // 2 + dy,
    }
    return crop_aligned, webcam_640x480, aug_params


def get_full_prototype_ranking(feat_norm, multi_prototypes, classes):
    """Return full per-class prototype similarity ranking."""
    sims = np.einsum('ckd,d->ck', multi_prototypes, feat_norm)
    class_max_sims = np.max(sims, axis=1)
    ranking_indices = np.argsort(class_max_sims)[::-1]
    ranking = []
    for idx in ranking_indices:
        ranking.append({
            "class": classes[idx],
            "similarity": round(float(class_max_sims[idx]), 4)
        })
    return ranking


def run_targeted_analysis():
    print("=" * 110)
    print("TARGETED FAILURE ANALYSIS FOR WEAK WEBCAM ASL CLASSES")
    print(f"Focus classes: {', '.join(WEAK_CLASSES)}")
    print("=" * 110)

    predictor.load_model()
    assert predictor.model is not None, "Model failed to load!"
    assert predictor.feat_model is not None, "Feature model failed to load!"
    assert predictor.multi_prototypes is not None, "Multi-prototypes not loaded!"

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    os.makedirs(GALLERY_DIR, exist_ok=True)

    # Storage for ALL 870 frames (all 29 classes)
    all_records = []
    per_class_records = defaultdict(list)

    np.random.seed(42)

    print(f"\n[*] Processing all {len(classes) * FRAMES_PER_CLASS} frames...")

    for class_idx, cls in enumerate(classes):
        all_class_files = sorted(glob.glob(os.path.join(TRAIN_DIR, cls, "*.jpg")))
        sample_files = all_class_files[2000:2000 + FRAMES_PER_CLASS]
        if len(sample_files) < FRAMES_PER_CLASS:
            sample_files = all_class_files[-FRAMES_PER_CLASS:]

        for frame_idx, fpath in enumerate(sample_files):
            im = Image.open(fpath).convert("RGB")
            crop_aligned, full_frame, aug_params = synthesize_webcam_frame(im, cls, frame_idx)

            # Get JPEG bytes (same as benchmark)
            buf = io.BytesIO()
            crop_aligned.save(buf, format="JPEG", quality=95)
            img_bytes = buf.getvalue()

            # Predict
            res = predictor.predict_image_bytes(img_bytes)

            pred = res["prediction"]
            conf = res["confidence"]
            is_unc = res["is_uncertain"]
            raw_top = res["top_predictions"][0]["class"] if res["top_predictions"] else "nothing"
            top_preds = res["top_predictions"]

            # Full prototype ranking
            img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_tensor, _ = preprocess_image(img_pil)
            feat = predictor.feat_model(img_tensor).numpy()[0]
            feat_norm = feat / (np.linalg.norm(feat) + 1e-7)
            proto_ranking = get_full_prototype_ranking(feat_norm, predictor.multi_prototypes, classes)

            hp = res.get("hand_presence", {})
            qm = res.get("quality_metrics", {})

            # Determine correctness
            if cls == "nothing":
                is_correct = (pred == "nothing")
            elif is_unc:
                is_correct = False
            else:
                is_correct = (pred == cls)

            record = {
                "frame_id": f"{cls}_{frame_idx:02d}",
                "true_class": cls,
                "file_source": os.path.basename(fpath),
                "predicted_class": pred,
                "raw_top_class": raw_top,
                "is_correct": is_correct,
                "confidence": conf,
                "top5_predictions": top_preds[:5],
                "is_uncertain": is_unc,
                "uncertainty_reason": res.get("uncertainty_reason"),
                "prototype_ranking_top5": proto_ranking[:5],
                "prototype_match": proto_ranking[0]["class"],
                "prototype_similarity": proto_ranking[0]["similarity"],
                "proto_rank_of_true_class": next(
                    (i for i, r in enumerate(proto_ranking) if r["class"] == cls), -1
                ),
                "proto_sim_of_true_class": next(
                    (r["similarity"] for r in proto_ranking if r["class"] == cls), 0.0
                ),
                "hand_presence_status": hp.get("status"),
                "hand_score": hp.get("hand_score"),
                "fg_ratio": hp.get("fg_ratio"),
                "cnt_ratio": hp.get("cnt_ratio"),
                "edge_density": hp.get("edge_density"),
                "quality_sharpness": qm.get("sharpness"),
                "quality_brightness": qm.get("brightness"),
                "quality_contrast": qm.get("contrast"),
                **aug_params,
            }

            all_records.append(record)
            per_class_records[cls].append(record)

            # Save gallery images for weak classes
            if cls in WEAK_CLASSES:
                cls_dir = os.path.join(GALLERY_DIR, cls)
                os.makedirs(cls_dir, exist_ok=True)
                status = "correct" if is_correct else ("uncertain" if is_unc else "incorrect")
                save_name = f"{status}_{frame_idx:02d}_{pred}_{conf:.0f}.jpg"
                crop_aligned.save(os.path.join(cls_dir, save_name), "JPEG", quality=95)

        print(f"  [{class_idx+1:2d}/29] {cls:>8}: processed {FRAMES_PER_CLASS} frames")

    # === ANALYSIS ===
    print("\n" + "=" * 110)
    print("ANALYSIS RESULTS")
    print("=" * 110)

    # Overall stats
    total = len(all_records)
    correct = sum(1 for r in all_records if r["is_correct"])
    incorrect = sum(1 for r in all_records if not r["is_correct"] and not r["is_uncertain"])
    uncertain = sum(1 for r in all_records if r["is_uncertain"])

    print(f"\nTotal frames: {total}")
    print(f"Correct: {correct} ({correct/total*100:.2f}%)")
    print(f"Incorrect: {incorrect}")
    print(f"Uncertain: {uncertain}")

    # Per-class breakdown
    per_class_summary = {}
    for cls in classes:
        recs = per_class_records[cls]
        c = sum(1 for r in recs if r["is_correct"])
        inc = sum(1 for r in recs if not r["is_correct"] and not r["is_uncertain"])
        unc = sum(1 for r in recs if r["is_uncertain"])
        per_class_summary[cls] = {
            "total": len(recs),
            "correct": c,
            "incorrect": inc,
            "uncertain": unc,
            "accuracy": round(c / len(recs) * 100, 1) if recs else 0.0
        }

    print(f"\n{'Class':<8} | {'Correct':>8} | {'Incorrect':>9} | {'Uncertain':>9} | {'Accuracy':>8}")
    print("-" * 55)
    for cls in classes:
        s = per_class_summary[cls]
        tag = " ** WEAK" if cls in WEAK_CLASSES else ""
        print(f"{cls:<8} | {s['correct']:>5}/{s['total']:<2} | {s['incorrect']:>9} | {s['uncertain']:>9} | {s['accuracy']:>6.1f}%{tag}")

    # Confusion groups for weak classes
    confusion_groups = defaultdict(int)
    weak_failures = defaultdict(list)

    for cls in WEAK_CLASSES:
        for r in per_class_records[cls]:
            if not r["is_correct"]:
                if r["is_uncertain"]:
                    key = f"{cls} -> uncertain"
                else:
                    key = f"{cls} -> {r['predicted_class']}"
                confusion_groups[key] += 1
                weak_failures[cls].append(r)

    print("\n" + "=" * 110)
    print("CONFUSION GROUPS FOR WEAK CLASSES")
    print("=" * 110)
    for pair, count in sorted(confusion_groups.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pair:<25} : {count:2d} frames")

    # Deep analysis: Prototype rank of true class for failures
    print("\n" + "=" * 110)
    print("PROTOTYPE ANALYSIS: WHERE DOES THE TRUE CLASS RANK?")
    print("=" * 110)
    for cls in WEAK_CLASSES:
        failures = weak_failures[cls]
        if not failures:
            continue
        ranks = [r["proto_rank_of_true_class"] for r in failures]
        sims = [r["proto_sim_of_true_class"] for r in failures]
        proto_top1_agrees_with_pred = sum(
            1 for r in failures if r["prototype_match"] == r["predicted_class"]
        )
        proto_top1_is_true = sum(
            1 for r in failures if r["prototype_match"] == cls
        )
        print(f"\n  {cls} ({len(failures)} failures):")
        print(f"    Prototype top-1 agrees with wrong CNN prediction: {proto_top1_agrees_with_pred}/{len(failures)}")
        print(f"    Prototype top-1 IS the true class:                {proto_top1_is_true}/{len(failures)}")
        print(f"    Mean prototype rank of true class:                {np.mean(ranks):.1f}")
        print(f"    Mean prototype similarity of true class:          {np.mean(sims):.4f}")
        print(f"    Rank distribution: {dict(zip(*np.unique(ranks, return_counts=True)))}")

    # CNN analysis: Where does true class sit in top-5?
    print("\n" + "=" * 110)
    print("CNN ANALYSIS: TRUE CLASS IN TOP-5 FOR FAILURES")
    print("=" * 110)
    for cls in WEAK_CLASSES:
        failures = weak_failures[cls]
        if not failures:
            continue
        true_in_top2 = 0
        true_in_top3 = 0
        true_in_top5 = 0
        for r in failures:
            top_classes = [p["class"] for p in r["top5_predictions"]]
            if cls in top_classes[:2]:
                true_in_top2 += 1
            if cls in top_classes[:3]:
                true_in_top3 += 1
            if cls in top_classes[:5]:
                true_in_top5 += 1
        print(f"  {cls} ({len(failures)} failures): true in top-2: {true_in_top2}, top-3: {true_in_top3}, top-5: {true_in_top5}")

    # Correct vs Incorrect image metrics comparison for weak classes
    print("\n" + "=" * 110)
    print("CORRECT vs INCORRECT FRAME STATISTICS (WEAK CLASSES)")
    print("=" * 110)
    for cls in WEAK_CLASSES:
        correct_recs = [r for r in per_class_records[cls] if r["is_correct"]]
        failure_recs = weak_failures[cls]
        if not correct_recs or not failure_recs:
            continue
        print(f"\n  {cls}:")
        for metric in ["confidence", "quality_sharpness", "quality_brightness", "quality_contrast",
                        "fg_ratio", "cnt_ratio", "edge_density", "hand_score"]:
            c_vals = [r[metric] for r in correct_recs if r.get(metric) is not None]
            f_vals = [r[metric] for r in failure_recs if r.get(metric) is not None]
            if c_vals and f_vals:
                print(f"    {metric:>20}: correct={np.mean(c_vals):.2f} ± {np.std(c_vals):.2f}  |  "
                      f"failure={np.mean(f_vals):.2f} ± {np.std(f_vals):.2f}")

    # Save JSON
    output_data = {
        "total_frames": total,
        "total_correct": correct,
        "total_incorrect": incorrect,
        "total_uncertain": uncertain,
        "raw_accuracy_pct": round(correct / total * 100, 2),
        "per_class_summary": per_class_summary,
        "confusion_groups": dict(sorted(confusion_groups.items(), key=lambda x: x[1], reverse=True)),
        "weak_class_failures": {
            cls: weak_failures[cls] for cls in WEAK_CLASSES
        },
        "weak_class_prototype_analysis": {},
    }

    for cls in WEAK_CLASSES:
        failures = weak_failures[cls]
        if not failures:
            output_data["weak_class_prototype_analysis"][cls] = {"failures": 0}
            continue
        ranks = [r["proto_rank_of_true_class"] for r in failures]
        sims = [r["proto_sim_of_true_class"] for r in failures]
        output_data["weak_class_prototype_analysis"][cls] = {
            "failures": len(failures),
            "proto_agrees_with_wrong_pred": sum(1 for r in failures if r["prototype_match"] == r["predicted_class"]),
            "proto_is_true_class": sum(1 for r in failures if r["prototype_match"] == cls),
            "mean_proto_rank_of_true": round(float(np.mean(ranks)), 2),
            "mean_proto_sim_of_true": round(float(np.mean(sims)), 4),
        }

    with open(ANALYSIS_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\n[+] JSON report saved to: {ANALYSIS_JSON}")

    # Save TXT
    with open(ANALYSIS_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("TARGETED FAILURE ANALYSIS FOR WEAK WEBCAM ASL CLASSES\n")
        f.write(f"Focus: {', '.join(WEAK_CLASSES)}\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Total frames: {total}\n")
        f.write(f"Correct: {correct} ({correct/total*100:.2f}%)\n")
        f.write(f"Incorrect: {incorrect}\n")
        f.write(f"Uncertain: {uncertain}\n\n")

        f.write("PER-CLASS ACCURACY:\n")
        f.write("-" * 60 + "\n")
        for cls in classes:
            s = per_class_summary[cls]
            tag = " ** WEAK" if cls in WEAK_CLASSES else ""
            f.write(f"  {cls:<8} {s['correct']:>2}/{s['total']:<2} ({s['accuracy']:>5.1f}%){tag}\n")

        f.write("\n\nCONFUSION GROUPS (WEAK CLASSES):\n")
        f.write("-" * 60 + "\n")
        for pair, count in sorted(confusion_groups.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {pair:<25} : {count:2d}\n")

        f.write("\n\nDETAILED FAILURE RECORDS:\n")
        f.write("-" * 100 + "\n")
        for cls in WEAK_CLASSES:
            f.write(f"\n--- {cls} FAILURES ---\n")
            for r in weak_failures[cls]:
                f.write(f"  {r['frame_id']}: true={r['true_class']}, pred={r['predicted_class']}, "
                        f"conf={r['confidence']:.1f}%, "
                        f"proto_match={r['prototype_match']}, proto_sim={r['prototype_similarity']:.4f}, "
                        f"proto_rank_true={r['proto_rank_of_true_class']}, "
                        f"proto_sim_true={r['proto_sim_of_true_class']:.4f}\n")
                f.write(f"           top5: {[(p['class'], p['confidence']) for p in r['top5_predictions']]}\n")
                f.write(f"           proto_top5: {[(p['class'], p['similarity']) for p in r['prototype_ranking_top5']]}\n")

    print(f"[+] TXT report saved to: {ANALYSIS_TXT}")
    print(f"[+] Failure galleries saved to: {GALLERY_DIR}/")
    print("=" * 110)


if __name__ == "__main__":
    run_targeted_analysis()
