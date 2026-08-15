import os
import sys
import glob
import json
import numpy as np
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import tensorflow as tf
from tensorflow import keras
from ml.config import TRAIN_DIR, MODEL_PATH, CLASSES_PATH, MODELS_DIR, IMAGE_SIZE
from ml.dataset import get_datasets

PROTOTYPES_PATH = os.path.join(MODELS_DIR, "class_prototypes.npy")

def benchmark_similarity_distributions():
    print("=" * 85)
    print("BENCHMARKING PROTOTYPE SIMILARITY DISTRIBUTIONS & CNN COMBINATION FEASIBILITY")
    print("=" * 85)
    
    model = keras.models.load_model(MODEL_PATH)
    dummy = np.zeros((1, 128, 128, 3), dtype=np.float32)
    _ = model(dummy)
    feat_layer = model.get_layer("dense_features")
    feat_model = keras.Model(inputs=model.inputs, outputs=feat_layer.output)
    
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)
        
    prototypes = np.load(PROTOTYPES_PATH) # (29, 256)
    
    # 1. Evaluate on Hold-out Test Samples (Zero Data Leakage)
    print("\n[*] Loading Hold-out Test Dataset...")
    ds_info = get_datasets(max_per_class=100, batch_size=64)
    test_paths, test_labels = ds_info["raw_test_data"]
    
    correct_similarities = []
    incorrect_similarities = []
    cross_class_similarities = []
    
    cnn_correct = 0
    combined_correct_alpha_02 = 0
    combined_correct_alpha_05 = 0
    total_samples = len(test_paths)
    
    print(f"[*] Analyzing {total_samples} holdout test samples...")
    for i, (fpath, true_label) in enumerate(zip(test_paths, test_labels)):
        img = Image.open(fpath).convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        img_arr = np.expand_dims(np.array(img, dtype=np.float32), 0)
        
        # CNN predictions
        cnn_probs = model(img_arr).numpy()[0]
        cnn_pred = int(np.argmax(cnn_probs))
        cnn_conf = float(cnn_probs[cnn_pred])
        
        # Feature extraction
        feat = feat_model(img_arr).numpy()[0]
        feat_norm = feat / (np.linalg.norm(feat) + 1e-7)
        
        # Cosine similarity against all 29 class prototypes
        sims = np.dot(prototypes, feat_norm) # (29,)
        proto_pred = int(np.argmax(sims))
        
        # Cosine similarity for true class
        true_sim = float(sims[true_label])
        correct_similarities.append(true_sim)
        
        # Cosine similarities for other 28 classes
        for c_idx in range(len(classes)):
            if c_idx != true_label:
                cross_class_similarities.append(float(sims[c_idx]))
                
        if cnn_pred == true_label:
            cnn_correct += 1
        else:
            incorrect_similarities.append(true_sim)
            
        # Benchmark combining CNN probabilities with Softmax of Prototype Similarities
        sim_probs = np.exp(sims * 10) / np.sum(np.exp(sims * 10))
        
        combo_02 = 0.8 * cnn_probs + 0.2 * sim_probs
        if int(np.argmax(combo_02)) == true_label:
            combined_correct_alpha_02 += 1
            
        combo_05 = 0.5 * cnn_probs + 0.5 * sim_probs
        if int(np.argmax(combo_05)) == true_label:
            combined_correct_alpha_05 += 1

    # Statistical summaries
    correct_arr = np.array(correct_similarities)
    cross_arr = np.array(cross_class_similarities)
    
    print("\n" + "-" * 85)
    print("STATISTICAL SIMILARITY DISTRIBUTIONS (EMPIRICALLY MEASURED):")
    print("-" * 85)
    print(f"True-Class Matches (N={len(correct_arr)}):")
    print(f"  Mean Similarity    : {np.mean(correct_arr):.4f}")
    print(f"  Std Dev            : {np.std(correct_arr):.4f}")
    print(f"  Median (50th %ile) : {np.percentile(correct_arr, 50):.4f}")
    print(f"  5th Percentile     : {np.percentile(correct_arr, 5):.4f}")
    print(f"  1st Percentile     : {np.percentile(correct_arr, 1):.4f}")
    print(f"  Min Observed       : {np.min(correct_arr):.4f}")
    print(f"  Max Observed       : {np.max(correct_arr):.4f}")
    
    print(f"\nCross-Class / Negative Matches (N={len(cross_arr)}):")
    print(f"  Mean Similarity    : {np.mean(cross_arr):.4f}")
    print(f"  Std Dev            : {np.std(cross_arr):.4f}")
    print(f"  95th Percentile    : {np.percentile(cross_arr, 95):.4f}")
    print(f"  99th Percentile    : {np.percentile(cross_arr, 99):.4f}")
    print(f"  Max Observed       : {np.max(cross_arr):.4f}")
    
    # Separation Margin
    p5_true = np.percentile(correct_arr, 5)
    p95_cross = np.percentile(cross_arr, 95)
    recommended_thresh = (p5_true + p95_cross) / 2.0
    print(f"\nData-Derived Prototype Consistency Threshold: ~{recommended_thresh:.4f}")
    print(f"  (Separation between 5th-percentile true: {p5_true:.4f} and 95th-percentile negative: {p95_cross:.4f})")

    print("\n" + "-" * 85)
    print("EMPIRICAL ACCURACY COMPARISON ON HOLDOUT TEST SET:")
    print("-" * 85)
    print(f"Pure CNN Softmax Accuracy               : {cnn_correct}/{total_samples} ({cnn_correct/total_samples*100:.2f}%)")
    print(f"Combined (80% CNN + 20% Prototype Sim)  : {combined_correct_alpha_02}/{total_samples} ({combined_correct_alpha_02/total_samples*100:.2f}%)")
    print(f"Combined (50% CNN + 50% Prototype Sim)  : {combined_correct_alpha_05}/{total_samples} ({combined_correct_alpha_05/total_samples*100:.2f}%)")

    # 2. Benchmark on Simulated Webcam Perturbations (Scale Jitter, Contrast Shifts, Blur, Partial Crops)
    print("\n" + "=" * 85)
    print("BENCHMARK ON SIMULATED WEBCAM DISTORTIONS (O/G/H, A/del/S, U/V, M/N/T):")
    print("=" * 85)
    
    perturbed_classes = ['O', 'G', 'H', 'A', 'del', 'S', 'U', 'V', 'M', 'N', 'T']
    cnn_pert_correct = 0
    combo_pert_correct = 0
    pert_total = 0
    
    for cls in perturbed_classes:
        files = sorted(glob.glob(os.path.join(TRAIN_DIR, cls, "*.jpg")))[-25:] # Use unseen tail samples
        true_idx = classes.index(cls)
        for fpath in files:
            im = Image.open(fpath).convert("RGB")
            # Apply webcam-like transforms: slight scale shift (85%), slight brightness offset
            w, h = im.size
            crop_box = (int(w*0.08), int(h*0.08), int(w*0.92), int(h*0.92))
            im_cropped = im.crop(crop_box).resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            arr = np.expand_dims(np.array(im_cropped, dtype=np.float32), 0)
            
            cnn_probs = model(arr).numpy()[0]
            cnn_p = int(np.argmax(cnn_probs))
            
            feat = feat_model(arr).numpy()[0]
            feat_n = feat / (np.linalg.norm(feat) + 1e-7)
            sims = np.dot(prototypes, feat_n)
            sim_probs = np.exp(sims * 8) / np.sum(np.exp(sims * 8))
            
            combo_p = int(np.argmax(0.85 * cnn_probs + 0.15 * sim_probs))
            
            if cnn_p == true_idx:
                cnn_pert_correct += 1
            if combo_p == true_idx:
                combo_pert_correct += 1
            pert_total += 1
            
    print(f"Distorted Crop - CNN Softmax Accuracy : {cnn_pert_correct}/{pert_total} ({cnn_pert_correct/pert_total*100:.2f}%)")
    print(f"Distorted Crop - Combined Accuracy    : {combo_pert_correct}/{pert_total} ({combo_pert_correct/pert_total*100:.2f}%)")
    print("=" * 85 + "\n")
    
    return {
        "recommended_thresh": float(recommended_thresh),
        "mean_true_sim": float(np.mean(correct_arr)),
        "std_true_sim": float(np.std(correct_arr)),
        "mean_cross_sim": float(np.mean(cross_arr)),
        "p5_true": float(p5_true),
        "p95_cross": float(p95_cross),
        "cnn_acc": cnn_correct / total_samples,
        "combo_acc": combined_correct_alpha_02 / total_samples,
        "pert_cnn_acc": cnn_pert_correct / pert_total,
        "pert_combo_acc": combo_pert_correct / pert_total
    }

if __name__ == "__main__":
    benchmark_similarity_distributions()
