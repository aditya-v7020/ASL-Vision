from pathlib import Path

# Robust Project Root Directory (independent of current working directory / environment)
BASE_DIR = Path(__file__).resolve().parent.parent

# Absolute Paths for Model & Artifacts
MODELS_DIR = BASE_DIR / "ml" / "models"
MODEL_PATH = str(MODELS_DIR / "sign_language_model.keras")
CLASSES_PATH = str(MODELS_DIR / "classes.json")
PROTOTYPES_PATH = str(MODELS_DIR / "class_prototypes.npy")
METADATA_PATH = str(MODELS_DIR / "reference_metadata.json")
MULTI_PROTOTYPES_PATH = str(MODELS_DIR / "class_multi_prototypes.npy")
MULTI_METADATA_PATH = str(MODELS_DIR / "multi_reference_metadata.json")

IMAGE_SIZE = (128, 128)

CORS_ORIGINS = [
    "https://asl-vision-xi.vercel.app",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

