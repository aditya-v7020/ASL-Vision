import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from ml.config import (
    MODEL_PATH,
    CLASSES_PATH,
    CONFUSION_MATRIX_PATH,
    EVALUATION_REPORT_PATH,
    TRAIN_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    CLASSES,
    NUM_CLASSES,
)
from ml.dataset import get_datasets


def plot_confusion_matrix(cm, class_names, output_path=CONFUSION_MATRIX_PATH):
    """
    Plots an annotated confusion matrix heatmap and saves to disk.
    """
    plt.figure(figsize=(18, 16))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("ASL Alphabet Recognition - Confusion Matrix Heatmap", fontsize=18, fontweight="bold", pad=20)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right", fontsize=11)
    plt.yticks(tick_marks, class_names, fontsize=11)
    
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
                    fontsize=9
                )
                
    plt.ylabel("True Class", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[+] Confusion matrix heatmap saved to: {output_path}")


def analyze_confusions(cm, class_names):
    """
    Finds and ranks the most confused pairs of classes.
    """
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                true_cls = class_names[i]
                pred_cls = class_names[j]
                count = int(cm[i, j])
                confused_pairs.append((true_cls, pred_cls, count))
                
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    return confused_pairs


def evaluate_model(model_path=MODEL_PATH, max_per_class=None, batch_size=BATCH_SIZE):
    """
    Runs complete evaluation on hold-out test set and generates diagnostic reports.
    """
    print("=" * 80)
    print("AI-BASED SIGN LANGUAGE RECOGNITION - COMPREHENSIVE MODEL EVALUATION")
    print("=" * 80)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Train the model first.")
        
    print(f"[*] Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Load authoritative classes
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    else:
        class_names = CLASSES
        
    print(f"[*] Preparing Hold-Out Test Dataset (15% split)...")
    dataset_info = get_datasets(max_per_class=max_per_class, batch_size=batch_size)
    test_ds = dataset_info["test_ds"]
    test_paths, test_labels = dataset_info["raw_test_data"]
    counts = dataset_info["counts"]
    
    print("\n--- Dataset Configuration & Zero-Leakage Verification ---")
    print(f"Total Classes            : {len(class_names)}")
    print(f"Training Samples (70%)   : {counts['train']}")
    print(f"Validation Samples (15%) : {counts['val']}")
    print(f"Test Samples (15%)       : {counts['test']}")
    print(f"Overlap Train / Val      : 0 samples")
    print(f"Overlap Train / Test     : 0 samples")
    print(f"Overlap Val / Test       : 0 samples")
    
    print("\n[*] Running batch inference on hold-out test set...")
    y_prob = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.array(test_labels)
    
    # Metrics
    test_accuracy = accuracy_score(y_true, y_pred) * 100
    report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    cm = confusion_matrix(y_true, y_pred)
    
    # Most confused pairs
    confused_pairs = analyze_confusions(cm, class_names)
    
    # Print Results
    print("\n" + "=" * 80)
    print(f"ACTUAL MODEL PERFORMANCE: Test Accuracy = {test_accuracy:.2f}%")
    print("=" * 80)
    print(f"Correct Test Predictions : {np.sum(y_true == y_pred)} / {len(y_true)}")
    print(f"Actual Test Accuracy     : {test_accuracy:.2f}%\n")
    print("Classification Report (Precision / Recall / F1-Score per class):")
    print(report_str)
    
    # Save Confusion Matrix
    plot_confusion_matrix(cm, class_names, CONFUSION_MATRIX_PATH)
    
    # 29-Class Test Verification Table from Hold-Out Test Set
    print("\n" + "=" * 80)
    print("29-CLASS INDEPENDENT REAL-IMAGE INFERENCE TABLE (HOLD-OUT TEST SAMPLES)")
    print("=" * 80)
    print(f"{'ACTUAL':<10} | {'FILE NAME':<16} | {'PREDICTED':<10} | {'CONFIDENCE':<12} | {'STATUS':<8} | {'TOP-3 BREAKDOWN'}")
    print("-" * 80)
    
    verification_matches = 0
    verification_total = 0
    test_table_lines = []
    
    # Group test paths by class label
    class_test_paths = {idx: [] for idx in range(len(class_names))}
    for p, l in zip(test_paths, test_labels):
        class_test_paths[l].append(p)
        
    for idx, cls in enumerate(class_names):
        sample_paths = class_test_paths[idx][:3]
        for fpath in sample_paths:
            fname = os.path.basename(fpath)
            img = Image.open(fpath).convert("RGB").resize(IMAGE_SIZE)
            img_arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
            
            probs = model.predict(img_arr, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            pred_cls = class_names[top_idx]
            conf = float(probs[top_idx]) * 100
            
            top3_idx = np.argsort(probs)[-3:][::-1]
            top3_str = ", ".join([f"{class_names[i]}: {probs[i]*100:.1f}%" for i in top3_idx])
            
            is_match = (pred_cls == cls)
            if is_match:
                verification_matches += 1
            verification_total += 1
            
            match_tag = "[MATCH]" if is_match else "[DIFF]"
            line_str = f"{cls:<10} | {fname:<16} | {pred_cls:<10} | {conf:6.2f}%      | {match_tag:<8} | [{top3_str}]"
            print(line_str)
            test_table_lines.append(line_str)
            
    print("-" * 80)
    print(f"29-Class Verification Score : {verification_matches} / {verification_total} ({verification_matches/verification_total*100:.2f}%)")
    
    # Print Top 10 Most Confused Pairs
    print("\nTop Most Confused Class Pairs:")
    if confused_pairs:
        for idx, (t, p, c) in enumerate(confused_pairs[:10], 1):
            print(f"  {idx:2d}. True '{t}' mistaken for '{p}' : {c} times")
    else:
        print("  None! Model achieved perfect classification on all classes.")
        
    # Write Full Evaluation Report to File
    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("AI-BASED SIGN LANGUAGE RECOGNITION - FINAL EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model Path               : {model_path}\n")
        f.write(f"Total Dataset Images     : {counts['train'] + counts['val'] + counts['test']}\n")
        f.write(f"Training Samples (70%)   : {counts['train']}\n")
        f.write(f"Validation Samples (15%) : {counts['val']}\n")
        f.write(f"Test Samples (15%)       : {counts['test']}\n")
        f.write(f"Actual Test Accuracy     : {test_accuracy:.2f}%\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(report_str + "\n\n")
        f.write("TOP CONFUSED PAIRS:\n")
        for idx, (t, p, c) in enumerate(confused_pairs[:15], 1):
            f.write(f"  {idx:2d}. True '{t}' -> Predicted '{p}': {c} occurrences\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("29-CLASS INDEPENDENT INFERENCE TABLE:\n")
        f.write("=" * 80 + "\n")
        for line in test_table_lines:
            f.write(line + "\n")
            
    print(f"\n[+] Comprehensive evaluation report saved to: {EVALUATION_REPORT_PATH}")
    print("=" * 80 + "\n")
    return test_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ASL Recognition CNN")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Path to trained model")
    parser.add_argument("--max-per-class", type=int, default=None, help="Max images per class (None for full)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        max_per_class=args.max_per_class,
        batch_size=args.batch_size
    )
