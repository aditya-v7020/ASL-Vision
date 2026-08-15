import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(BASE_DIR, "ml", "models", "sign_language_model.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "ml", "models", "classes.json")
PROTOTYPES_PATH = os.path.join(BASE_DIR, "ml", "models", "class_prototypes.npy")
METADATA_PATH = os.path.join(BASE_DIR, "ml", "models", "reference_metadata.json")
MULTI_PROTOTYPES_PATH = os.path.join(BASE_DIR, "ml", "models", "class_multi_prototypes.npy")
MULTI_METADATA_PATH = os.path.join(BASE_DIR, "ml", "models", "multi_reference_metadata.json")

IMAGE_SIZE = (128, 128)

CORS_ORIGINS = [
    "https://asl-vision-xi.vercel.app",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

