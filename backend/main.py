import os
import sys

# Ensure root directory is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from backend.schemas import HealthResponse, PredictionResponse, ClassesResponse
from backend.predictor import predictor
from ml.config import TRAIN_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("backend.main")

# Allowed Origins for Production (Vercel) and Local Development
CORS_ORIGINS = [
    "https://asl-vision-xi.vercel.app",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: preload model
    logger.info("Initializing Sign Language Recognition Backend...")
    predictor.load_model()
    yield
    logger.info("Shutting down backend...")

app = FastAPI(
    title="AI-Based Sign Language Recognition API",
    description="Deep Learning CNN Backend for real-time ASL alphabet recognition",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["General"])
async def root():
    return {
        "message": "AI-Based Sign Language Recognition API is active.",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "classes": "/classes",
            "reference": "/reference/{class_name}",
            "predict": "/predict [POST]"
        }
    }

@app.get("/health", tags=["General"])
def health():
    return {
        "status": "ok",
        "service": "ASL-Vision"
    }

@app.get("/classes", response_model=ClassesResponse, tags=["Metadata"])
async def get_classes():
    return ClassesResponse(
        classes=predictor.classes,
        total=len(predictor.classes)
    )

@app.get("/reference/{class_name}", tags=["Metadata"])
async def get_reference_image(class_name: str):
    clean_cls = class_name.strip()
    cls_dir = os.path.join(TRAIN_DIR, clean_cls)
    if not os.path.exists(cls_dir):
        raise HTTPException(status_code=404, detail=f"Class '{clean_cls}' not found")
    ref_file = os.path.join(cls_dir, f"{clean_cls}1.jpg")
    if not os.path.exists(ref_file):
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            raise HTTPException(status_code=404, detail=f"No reference images for '{clean_cls}'")
        ref_file = os.path.join(cls_dir, files[0])
    return FileResponse(ref_file, media_type="image/jpeg")


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_sign(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file must be an image. Received content-type: {file.content_type}"
        )

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )
        if len(contents) > 10 * 1024 * 1024:  # 10 MB limit
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image file size exceeds maximum limit of 10MB."
            )

        result = predictor.predict_image_bytes(contents)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(re))
    except Exception as e:
        logger.error(f"Unexpected prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal prediction error.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
