import os
import sys
import asyncio
from io import BytesIO
from starlette.datastructures import UploadFile as StarletteUploadFile

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.main import root, health_check, get_classes, predict_sign
from backend.predictor import predictor
from ml.config import TRAIN_DIR, CLASSES

def test_api_handlers():
    print("=" * 80)
    print("TESTING FASTAPI BACKEND HANDLERS DIRECTLY")
    print("=" * 80)

    # Preload model
    predictor.load_model()

    # 1. Root
    root_res = asyncio.run(root())
    assert "message" in root_res
    print(f"[+] GET / -> {root_res['message']}")

    # 2. Health Check
    health_res = asyncio.run(health_check())
    assert health_res.status == "healthy"
    assert health_res.model_loaded is True
    assert health_res.classes_count == 29
    print(f"[+] GET /health -> Status: {health_res.status}, Model Loaded: {health_res.model_loaded}, Classes: {health_res.classes_count}")

    # 3. Classes List
    classes_res = asyncio.run(get_classes())
    assert classes_res.total == 29
    assert classes_res.classes == CLASSES
    print(f"[+] GET /classes -> Total: {classes_res.total} classes verified identical to authoritative list.")

    # 4. Predict Sign across diverse ASL signs
    test_signs = ["A", "B", "C", "G", "H", "O", "del", "nothing", "space", "V", "Y", "Z"]
    print("\n" + "-" * 80)
    print("TESTING /predict INFERENCE HANDLER ACROSS ASL SIGNS")
    print("-" * 80)
    
    matches = 0
    for sign in test_signs:
        img_path = os.path.join(TRAIN_DIR, sign, f"{sign}1.jpg")
        with open(img_path, "rb") as f:
            file_bytes = f.read()

        upload_file = StarletteUploadFile(file=BytesIO(file_bytes), filename=f"{sign}1.jpg", headers={"content-type": "image/jpeg"})
        pred_res = asyncio.run(predict_sign(upload_file))

        pred = pred_res["prediction"]
        conf = pred_res["confidence"]
        top3 = [f"{p['class']} ({p['confidence']}%)" for p in pred_res["top_predictions"]]
        is_match = (pred == sign)
        if is_match:
            matches += 1
        tag = "[MATCH]" if is_match else "[DIFF]"
        print(f"  Target: {sign:<8} | Predicted: {pred:<8} | Confidence: {conf:6.2f}% | Top-3: {top3} {tag}")

    print("-" * 80)
    print(f"[+] Backend Inference Verification: {matches} / {len(test_signs)} matched ({matches/len(test_signs)*100:.1f}%)")
    print("[+] All FastAPI Backend Handlers PASSED!\n")

if __name__ == "__main__":
    test_api_handlers()
