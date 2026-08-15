# AI-Based Sign Language Recognition System

A deep-learning-powered American Sign Language (ASL) Alphabet Recognition web application. The system uses a Convolutional Neural Network (CNN) trained with **TensorFlow and Keras**, a high-performance **FastAPI** backend, and a modern **React + Vite** frontend supporting real-time webcam prediction and manual image upload.

---

## System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                      React + Vite UI                        │
│   (Webcam Live Stream / Image Upload / Real-Time Metrics)   │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP Multipart / JSON
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                       FastAPI Backend                       │
│    (CORS, Image Preprocessing, In-Memory Model Predictor)   │
└──────────────────────────────┬──────────────────────────────┘
                               │ Tensor Normalization (128x128x3)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                TensorFlow + Keras CNN Model                 │
│    (Conv2D + BatchNorm + MaxPool + GlobalAvgPool + Dense)   │
│                 29 ASL Alphabet Sign Classes                │
└─────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

- **Deep Learning Framework:** TensorFlow 2.21.0 + Keras
- **Environment:** Python 3.12.10
- **Backend API:** FastAPI, Uvicorn, Pydantic, Python-Multipart
- **Image Processing & ML Utilities:** OpenCV, Pillow, Scikit-learn, NumPy, Matplotlib
- **Frontend Framework:** React 18, Vite, Lucide-React Icons, Axios
- **Design System:** Vanilla CSS with custom Glassmorphic dark mode, responsive grid, dynamic badges, and progress meters.

---

## Dataset Structure & Splitting Strategy

The application recognizes **29 classes** (`A`–`Z`, `del`, `nothing`, `space`):

```text
DeepLearning Project/
├── asl_alphabet_train/
│   └── asl_alphabet_train/
│       ├── A/ ... (3,000 images per class)
│       └── ... [29 classes, 87,000 total images]
└── asl_alphabet_test/
    └── asl_alphabet_test/
        ├── A_test.jpg ... [28 sample test images]
```

### Data Splitting Policy
1. **Original Dataset Untouched:** No files are moved, renamed, or modified.
2. **Stratified Partition:** Training data is split into:
   - **70% Training Set:** Used for gradient descent optimization with on-the-fly data augmentation.
   - **15% Validation Set:** Used during training for early stopping and learning rate scheduling.
   - **15% Test Set:** Hold-out set used strictly for final evaluation with ground-truth labels.
3. **Zero Data Leakage:** Reproducible seed ensures zero overlap (`Train ∩ Val = ∅`, `Train ∩ Test = ∅`, `Val ∩ Test = ∅`).
4. **Out-of-Sample Test Evaluation:** Sample images in `asl_alphabet_test/` are also evaluated as visual validation samples.

---

## Project Directory Tree

```text
DeepLearning Project/
├── .venv/                      # Dedicated Python 3.12 virtual environment
├── asl_alphabet_train/         # Extracted ASL Training Dataset (29 folders)
├── asl_alphabet_test/          # Extracted ASL Test Dataset (28 sample images)
├── ml/                         # Machine Learning Module
│   ├── config.py               # Paths, hyperparameters, class labels
│   ├── dataset.py              # Stratified split, tf.data pipeline & augmentation
│   ├── model.py                # TensorFlow Keras CNN architecture
│   ├── train.py                # Training script with callbacks & history plotting
│   ├── evaluate.py             # Evaluation script (Accuracy, Report, Confusion Matrix)
│   ├── predict.py              # CLI inference tool
│   ├── models/                 # Saved models & classes.json
│   │   ├── sign_language_model.keras
│   │   └── classes.json
│   └── outputs/                # Training plots & evaluation reports
│       ├── training_history.png
│       ├── confusion_matrix.png
│       └── evaluation_results.txt
├── backend/                    # FastAPI Server
│   ├── main.py                 # API entrypoint & routes (/, /health, /classes, /predict)
│   ├── predictor.py            # Singleton model manager & tensor preprocessor
│   ├── schemas.py              # Pydantic schemas
│   ├── config.py               # Backend settings & CORS
│   └── requirements.txt        # Backend dependencies
├── frontend/                   # React + Vite Web Application
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx
│   │   │   ├── WebcamPredictor.jsx
│   │   │   ├── ImageUploadPredictor.jsx
│   │   │   └── PredictionResult.jsx
│   │   ├── services/
│   │   │   └── api.js
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
├── requirements.txt            # Consolidated Python dependencies
├── .gitignore
└── README.md
```

---

## Setup & Running Guide

### 1. Python Virtual Environment Setup

Activate the project Python 3.12 virtual environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Or run commands directly via .venv\Scripts\python.exe
```

Install/verify Python dependencies:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

### 2. Machine Learning Pipeline

#### Train the CNN Model
Train with fast development mode (200 images per class for fast verification):
```powershell
.venv\Scripts\python.exe ml/train.py --epochs 10 --batch-size 32 --max-per-class 200
```

To train on the full 87,000 images dataset:
```powershell
.venv\Scripts\python.exe ml/train.py --full-train --epochs 15 --batch-size 32
```

Outputs generated:
- Model saved to: `ml/models/sign_language_model.keras`
- Class mapping: `ml/models/classes.json`
- Training loss & accuracy curves: `ml/outputs/training_history.png`

#### Evaluate the Model
Run full evaluation on the hold-out test set:
```powershell
.venv\Scripts\python.exe ml/evaluate.py
```

Outputs generated:
- Confusion matrix heatmap: `ml/outputs/confusion_matrix.png`
- Evaluation report & per-class precision/recall: `ml/outputs/evaluation_results.txt`

#### Standalone CLI Prediction
Predict on any image path:
```powershell
.venv\Scripts\python.exe ml/predict.py "asl_alphabet_test/asl_alphabet_test/C_test.jpg"
```

---

### 3. Start the FastAPI Backend

Run the FastAPI server on port 8000:
```powershell
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

- Interactive API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

### 4. Start the React Frontend

Open a new terminal, navigate to `frontend/`, and launch the Vite dev server:
```powershell
cd frontend
npm install
npm run dev
```

Open your browser at: [http://127.0.0.1:5173](http://127.0.0.1:5173)

---

## API Documentation

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Welcome message and API overview |
| `/health` | `GET` | Health check returning status, framework, and loaded model state |
| `/classes` | `GET` | List of 29 supported ASL alphabet sign labels |
| `/predict` | `POST` | Accepts multipart image file (`file`), returns top predicted sign and confidence breakdown |

### Sample `POST /predict` Response:
```json
{
  "prediction": "C",
  "confidence": 97.59,
  "top_predictions": [
    {
      "class": "C",
      "confidence": 97.59
    },
    {
      "class": "D",
      "confidence": 1.24
    },
    {
      "class": "E",
      "confidence": 0.41
    }
  ]
}
```

---

## Usage Instructions

1. **Live Webcam Mode:**
   - Click **Start Camera** to allow browser camera access.
   - Position your hand sign inside the target guide box.
   - Click **Capture & Analyze** for instant recognition, or toggle **Auto-Stream Scan** for continuous real-time prediction.
2. **Image Upload Mode:**
   - Switch to the **Image File Upload** tab.
   - Drag and drop any ASL image or click to select a file from your machine.
   - Click **Analyze Image** to view the detected sign, confidence percentage, and top-3 breakdown.

# Code to run this project

# In Terminal 1, start the backend:

.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# In Terminal 2, start the frontend:

cd frontend
npm run dev
