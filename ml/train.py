import os
import sys

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import argparse
import matplotlib
matplotlib.use("Agg")  # Headless backend for saving plots
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from ml.config import (
    MODEL_PATH,
    CLASSES_PATH,
    HISTORY_PLOT_PATH,
    EPOCHS,
    BATCH_SIZE,
    MAX_IMAGES_PER_CLASS,
    IMAGE_SHAPE,
)
from ml.dataset import get_datasets
from ml.model import build_sign_language_model

def plot_and_save_history(history, output_path=HISTORY_PLOT_PATH):
    """
    Plots training & validation accuracy and loss side by side, then saves the figure.
    """
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", marker="o", color="#2563eb")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", marker="s", color="#16a34a")
    plt.title("Model Accuracy over Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", marker="o", color="#dc2626")
    plt.plot(epochs_range, val_loss, label="Validation Loss", marker="s", color="#ea580c")
    plt.title("Model Loss over Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[+] Saved training history plot to: {output_path}")


def train(epochs=EPOCHS, batch_size=BATCH_SIZE, max_per_class=MAX_IMAGES_PER_CLASS):
    """
    Executes the training workflow.
    """
    print("=" * 60)
    print("AI-Based Sign Language Recognition - Model Training")
    print("=" * 60)
    print(f"[*] Configuration: Epochs={epochs}, Batch Size={batch_size}, Max Imgs/Class={max_per_class}")
    
    # 1. Load Data
    print("\n[*] Preparing datasets (Train / Val / Test)...")
    dataset_info = get_datasets(max_per_class=max_per_class, batch_size=batch_size)
    
    train_ds = dataset_info["train_ds"]
    val_ds = dataset_info["val_ds"]
    class_names = dataset_info["class_names"]
    counts = dataset_info["counts"]
    
    print(f"[+] Total classes: {len(class_names)}")
    print(f"[+] Samples: Train={counts['train']} | Validation={counts['val']} | Test={counts['test']}")
    
    # 2. Save class names metadata for backend & predictor
    with open(CLASSES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"[+] Saved class names to: {CLASSES_PATH}")
    
    # 3. Build Model
    print("\n[*] Initializing CNN Model Architecture...")
    model = build_sign_language_model(
        input_shape=IMAGE_SHAPE,
        num_classes=len(class_names)
    )
    model.summary()
    
    # 4. Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # 5. Train
    print("\n[*] Starting Model Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # 6. Ensure best model is saved
    model.save(MODEL_PATH)
    print(f"\n[+] Trained model saved successfully to: {MODEL_PATH}")
    
    # 7. Generate and save training graphs
    plot_and_save_history(history, HISTORY_PLOT_PATH)
    
    final_train_acc = history.history["accuracy"][-1] * 100
    final_val_acc = history.history["val_accuracy"][-1] * 100
    print(f"\n[+] Training complete! Final Train Acc: {final_train_acc:.2f}%, Best Val Acc: {max(history.history['val_accuracy'])*100:.2f}%")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASL Alphabet CNN Model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--max-per-class", type=int, default=MAX_IMAGES_PER_CLASS, help="Max images per class (None for full dataset)")
    parser.add_argument("--full-train", action="store_true", help="Train on the entire 87,000 image dataset")
    
    args = parser.parse_args()
    
    max_imgs = None if args.full_train else args.max_per_class
    train(epochs=args.epochs, batch_size=args.batch_size, max_per_class=max_imgs)
