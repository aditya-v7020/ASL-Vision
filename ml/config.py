import os
import json

# Project Root Directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Dataset Paths
TRAIN_DIR = os.path.join(BASE_DIR, "asl_alphabet_train", "asl_alphabet_train")
TEST_DIR = os.path.join(BASE_DIR, "asl_alphabet_test", "asl_alphabet_test")

# Fallback paths if flat directory structure
if not os.path.exists(TRAIN_DIR):
    TRAIN_DIR = os.path.join(BASE_DIR, "asl_alphabet_train")
if not os.path.exists(TEST_DIR):
    TEST_DIR = os.path.join(BASE_DIR, "asl_alphabet_test")

# Artifacts & Output Paths
ML_DIR = os.path.join(BASE_DIR, "ml")
MODELS_DIR = os.path.join(ML_DIR, "models")
OUTPUTS_DIR = os.path.join(ML_DIR, "outputs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "sign_language_model.keras")
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.json")
HISTORY_PLOT_PATH = os.path.join(OUTPUTS_DIR, "training_history.png")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
EVALUATION_REPORT_PATH = os.path.join(OUTPUTS_DIR, "evaluation_results.txt")

# Image & Training Hyperparameters
IMAGE_SIZE = (128, 128)
IMAGE_SHAPE = (128, 128, 3)
BATCH_SIZE = 128
EPOCHS = 12
LEARNING_RATE = 0.001
RANDOM_SEED = 42

# Dataset Split Configuration: 70% Train, 15% Validation, 15% Test
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Full Dataset Training (87,000 total images across all 29 classes)
MAX_IMAGES_PER_CLASS = None

# Authoritative 29 ASL Alphabet Classes
CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]
NUM_CLASSES = len(CLASSES)

# Save classes.json immediately to guarantee synchronization
with open(CLASSES_PATH, "w", encoding="utf-8") as f:
    json.dump(CLASSES, f, indent=2)
