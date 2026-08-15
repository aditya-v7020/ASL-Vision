import os
import sys
import glob
import json
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import tensorflow as tf
from tensorflow import keras
from ml.config import TRAIN_DIR, MODEL_PATH, CLASSES_PATH, MODELS_DIR, IMAGE_SIZE, CLASSES

MULTI_PROTOTYPES_PATH = os.path.join(MODELS_DIR, "class_multi_prototypes.npy")
MULTI_METADATA_PATH = os.path.join(MODELS_DIR, "multi_reference_metadata.json")

def build_multi_prototypes(clusters_per_class=5, samples_per_class=200):
    print("=" * 85)
    print(f"BUILDING MULTI-PROTOTYPE REFERENCE EMBEDDINGS ({clusters_per_class} CLUSTERS/CLASS)")
    print("=" * 85)

    model = keras.models.load_model(MODEL_PATH)
    dummy = np.zeros((1, 128, 128, 3), dtype=np.float32)
    _ = model(dummy)
    feat_layer = model.get_layer("dense_features")
    feat_model = keras.Model(inputs=model.inputs, outputs=feat_layer.output)

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    # Multi-prototype array: (num_classes, clusters_per_class, 256)
    multi_proto_matrix = np.zeros((len(class_names), clusters_per_class, 256), dtype=np.float32)
    multi_stats = {}

    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(TRAIN_DIR, cls)
        all_files = sorted(glob.glob(os.path.join(cls_dir, "*.jpg")))
        # Use samples across the training range
        selected_files = all_files[:samples_per_class]

        imgs = []
        for fpath in selected_files:
            im = Image.open(fpath).convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            imgs.append(np.array(im, dtype=np.float32))

        batch_arr = np.array(imgs)
        feats = feat_model(batch_arr).numpy()
        
        # L2-normalize
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-7
        feats_norm = feats / norms

        # K-Means clustering in normalized feature space
        kmeans = KMeans(n_clusters=clusters_per_class, random_state=42, n_init=10)
        kmeans.fit(feats_norm)
        
        # Normalize cluster centers to unit sphere
        centers = kmeans.cluster_centers_
        centers_norm = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-7)

        multi_proto_matrix[idx] = centers_norm

        # Measure intra-cluster coverage
        max_sims = np.max(np.dot(feats_norm, centers_norm.T), axis=1)
        multi_stats[cls] = {
            "mean_max_sim": float(np.mean(max_sims)),
            "min_max_sim": float(np.min(max_sims)),
            "samples_used": len(selected_files)
        }
        print(f"Class {cls:<8} | Mean Best Similarity to Multi-Prototypes: {multi_stats[cls]['mean_max_sim']:.4f} (Min: {multi_stats[cls]['min_max_sim']:.4f})")

    np.save(MULTI_PROTOTYPES_PATH, multi_proto_matrix)
    print(f"\n[+] Saved multi-prototype matrix {multi_proto_matrix.shape} to: {MULTI_PROTOTYPES_PATH}")

    with open(MULTI_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "classes": class_names,
            "clusters_per_class": clusters_per_class,
            "feature_dim": 256,
            "class_stats": multi_stats
        }, f, indent=2)
    print(f"[+] Saved metadata to: {MULTI_METADATA_PATH}")
    print("=" * 85)
    return multi_proto_matrix

if __name__ == "__main__":
    build_multi_prototypes()
