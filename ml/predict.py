import os
import sys

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

from ml.config import MODEL_PATH, CLASSES_PATH, IMAGE_SIZE, CLASSES

def predict_single_image(image_path, model=None, class_names=None):
    """
    Runs inference on a single image file path and returns top predictions.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
        
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}. Train the model first.")
        model = keras.models.load_model(MODEL_PATH)
        
    if class_names is None:
        if os.path.exists(CLASSES_PATH):
            with open(CLASSES_PATH, "r") as f:
                class_names = json.load(f)
        else:
            class_names = CLASSES
            
    # Load and preprocess with PIL to match standard image inputs
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_tensor = np.expand_dims(img_array, axis=0)
    
    # Predict
    probabilities = model.predict(img_tensor, verbose=0)[0]
    
    # Top 3 predictions
    top_indices = np.argsort(probabilities)[::-1][:3]
    top_predictions = [
        {
            "class": class_names[idx],
            "confidence": round(float(probabilities[idx]) * 100, 2)
        }
        for idx in top_indices
    ]
    
    return {
        "prediction": top_predictions[0]["class"],
        "confidence": top_predictions[0]["confidence"],
        "top_predictions": top_predictions
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict ASL Sign from Image")
    parser.add_argument("image_path", type=str, help="Path to input image file")
    args = parser.parse_args()
    
    result = predict_single_image(args.image_path)
    print("\n--- Prediction Result ---")
    print(f"Predicted Sign : {result['prediction']}")
    print(f"Confidence     : {result['confidence']}%")
    print("\nTop 3 Predictions:")
    for p in result["top_predictions"]:
        print(f"  - {p['class']:<8} : {p['confidence']}%")
