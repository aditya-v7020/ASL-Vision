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
from ml.config import TRAIN_DIR, MODEL_PATH, CLASSES_PATH, MODELS_DIR, IMAGE_SIZE, CLASSES

PROTOTYPES_PATH = os.path.join(MODELS_DIR, "class_prototypes.npy")
METADATA_PATH = os.path.join(MODELS_DIR, "reference_metadata.json")

def extract_class_prototypes(samples_per_class=150):
    """
    Extracts normalized 256-dimensional feature representations for each of the 29 ASL classes
    from the trained CNN penultimate layer ('dense_features'), computing the centroid (mean prototype)
    and per-class intra-cluster dispersion metrics.
    """
    print("=" * 80)
    print("EXTRACTING CANONICAL ASL CLASS PROTOTYPES FROM DATASET")
    print("=" * 80)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
    print(f"[*] Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)
    
    # Ensure inputs are initialized in Keras 3
    dummy = np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)
    _ = model(dummy)
    
    feat_layer = model.get_layer("dense_features")
    feat_model = keras.Model(inputs=model.inputs, outputs=feat_layer.output)
    
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
        
    prototypes_dict = {}
    class_stats = {}
    
    for cls in class_names:
        cls_dir = os.path.join(TRAIN_DIR, cls)
        if not os.path.exists(cls_dir):
            print(f"[!] Warning: directory {cls_dir} does not exist!")
            continue
            
        all_files = sorted(glob.glob(os.path.join(cls_dir, "*.jpg")))
        selected_files = all_files[:samples_per_class]
        
        imgs = []
        for fpath in selected_files:
            im = Image.open(fpath).convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            imgs.append(np.array(im, dtype=np.float32))
            
        batch_arr = np.array(imgs)
        feats = feat_model(batch_arr).numpy()
        
        # L2-normalize individual feature vectors
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-7
        feats_norm = feats / norms
        
        # Calculate mean prototype vector and re-normalize to unit sphere
        mean_proto = np.mean(feats_norm, axis=0)
        mean_proto_norm = mean_proto / (np.linalg.norm(mean_proto) + 1e-7)
        
        # Compute intra-class similarity distribution (cosine similarities of class samples to their own prototype)
        intra_sims = np.dot(feats_norm, mean_proto_norm)
        
        prototypes_dict[cls] = mean_proto_norm
        class_stats[cls] = {
            "samples_used": len(selected_files),
            "intra_sim_mean": float(np.mean(intra_sims)),
            "intra_sim_std": float(np.std(intra_sims)),
            "intra_sim_min": float(np.min(intra_sims)),
            "intra_sim_max": float(np.max(intra_sims)),
            "reference_sample": os.path.basename(selected_files[0]) if selected_files else ""
        }
        
        print(f"Class {cls:<8} | Samples: {len(selected_files):3d} | Intra-sim Mean: {class_stats[cls]['intra_sim_mean']:.4f} (±{class_stats[cls]['intra_sim_std']:.4f})")
        
    # Save prototype matrix array matching exact class order
    proto_matrix = np.zeros((len(class_names), 256), dtype=np.float32)
    for idx, cls in enumerate(class_names):
        proto_matrix[idx] = prototypes_dict[cls]
        
    np.save(PROTOTYPES_PATH, proto_matrix)
    print(f"\n[+] Saved prototype matrix {proto_matrix.shape} to: {PROTOTYPES_PATH}")
    
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "classes": class_names,
            "feature_dim": 256,
            "samples_per_class": samples_per_class,
            "class_stats": class_stats
        }, f, indent=2)
    print(f"[+] Saved prototype metadata to: {METADATA_PATH}")
    print("=" * 80 + "\n")
    return proto_matrix, class_stats

if __name__ == "__main__":
    extract_class_prototypes(samples_per_class=150)
