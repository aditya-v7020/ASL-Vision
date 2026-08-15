import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ml.config import IMAGE_SHAPE, NUM_CLASSES, LEARNING_RATE

def build_sign_language_model(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE):
    """
    Builds an optimized deep Convolutional Neural Network (CNN) for 29-class ASL Alphabet recognition.
    
    Architecture:
    1. Input layer + Rescaling (0-255 -> 0-1)
    2. Stage 1: Conv2D(32, 3x3) + BatchNorm + ReLU + MaxPool(2x2) -> 64x64x32
    3. Stage 2: SeparableConv2D(64, 3x3) + BatchNorm + ReLU + MaxPool(2x2) -> 32x32x64
    4. Stage 3: SeparableConv2D(128, 3x3) + BatchNorm + ReLU + MaxPool(2x2) -> 16x16x128
    5. Stage 4: SeparableConv2D(256, 3x3) + BatchNorm + ReLU + MaxPool(2x2) -> 8x8x256
    6. Classification Head: GlobalAveragePooling2D + Dense(256, ReLU) + BatchNorm + Dropout(0.35) + Dense(num_classes, Softmax)
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape, name="input_image"),
        
        # Rescaling pixel values to [0, 1]
        layers.Rescaling(1.0 / 255.0, name="rescaling"),
        
        # Stage 1: 128x128 -> 64x64
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1"),
        layers.BatchNormalization(name="bn1"),
        layers.MaxPooling2D((2, 2), name="pool1"),
        
        # Stage 2: 64x64 -> 32x32
        layers.SeparableConv2D(64, (3, 3), padding="same", activation="relu", name="conv2"),
        layers.BatchNormalization(name="bn2"),
        layers.MaxPooling2D((2, 2), name="pool2"),
        
        # Stage 3: 32x32 -> 16x16
        layers.SeparableConv2D(128, (3, 3), padding="same", activation="relu", name="conv3"),
        layers.BatchNormalization(name="bn3"),
        layers.MaxPooling2D((2, 2), name="pool3"),
        
        # Stage 4: 16x16 -> 8x8
        layers.SeparableConv2D(256, (3, 3), padding="same", activation="relu", name="conv4"),
        layers.BatchNormalization(name="bn4"),
        layers.MaxPooling2D((2, 2), name="pool4"),
        
        # Classification Head
        layers.GlobalAveragePooling2D(name="global_avg_pool"),
        layers.Dense(256, activation="relu", name="dense_features"),
        layers.BatchNormalization(name="bn_dense"),
        layers.Dropout(0.35, name="dropout"),
        layers.Dense(num_classes, activation="softmax", name="predictions"),
    ], name="ASL_Alphabet_CNN")
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

if __name__ == "__main__":
    model = build_sign_language_model()
    model.summary()
