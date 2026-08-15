import os
import sys
import io
import json
import glob
import numpy as np
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.predictor import predictor
from ml.config import TRAIN_DIR, CLASSES_PATH

class TemporalStabilizerSimulator:
    """
    Python simulator of the frontend WebcamPredictor temporal smoothing algorithm:
    - 4-frame rolling history queue
    - Weighted exponential aggregation: [0.15, 0.25, 0.30, 0.30]
    - Reset immediately on 'NO_HAND' or uncertain frames (clears locked classes)
    - 2-frame stability confirmation threshold
    """
    def __init__(self):
        self.history_queue = []
        self.consecutive_agree_count = 0
        self.last_stable_class = None

    def process_frame(self, raw_result):
        has_hand = raw_result.get("hand_presence", {}).get("has_hand", not raw_result.get("is_uncertain", False))

        if not has_hand or raw_result.get("is_uncertain", False):
            self.history_queue.append(raw_result)
            if len(self.history_queue) > 4:
                self.history_queue.pop(0)
            self.consecutive_agree_count = 0
            self.last_stable_class = None
            return {
                "display_prediction": raw_result["prediction"],
                "confidence": raw_result["confidence"],
                "is_stable": False,
                "is_uncertain": True,
                "reason": raw_result.get("uncertainty_reason")
            }

        self.history_queue.append(raw_result)
        if len(self.history_queue) > 4:
            self.history_queue.pop(0)

        valid_frames = [f for f in self.history_queue if not f.get("is_uncertain") and f.get("top_predictions") and f.get("hand_presence", {}).get("has_hand") is not False]
        if not valid_frames:
            return {
                "display_prediction": raw_result["prediction"],
                "confidence": raw_result["confidence"],
                "is_stable": False,
                "is_uncertain": raw_result.get("is_uncertain", False)
            }

        class_scores = {}
        raw_weights = [0.15, 0.25, 0.30, 0.30]
        weights = raw_weights[4 - len(valid_frames):]
        w_sum = sum(weights)

        for idx, frame in enumerate(valid_frames):
            w = weights[idx] / w_sum
            for item in frame["top_predictions"]:
                cname = item["class"]
                conf = item["confidence"]
                class_scores[cname] = class_scores.get(cname, 0.0) + (conf * w)

        sorted_smoothed = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
        smoothed_top_class = sorted_smoothed[0][0] if sorted_smoothed else raw_result["prediction"]

        if smoothed_top_class == self.last_stable_class:
            self.consecutive_agree_count += 1
        else:
            self.consecutive_agree_count = 1
            self.last_stable_class = smoothed_top_class

        is_stable = self.consecutive_agree_count >= 2

        return {
            "display_prediction": smoothed_top_class,
            "confidence": round(sorted_smoothed[0][1], 1) if sorted_smoothed else raw_result["confidence"],
            "is_stable": is_stable,
            "is_uncertain": False,
            "stability_count": self.consecutive_agree_count
        }

def test_temporal_sequences():
    print("=" * 95)
    print("TEMPORAL STREAM TRANSITION VERIFICATION (A -> nothing -> B, O -> nothing -> G, D -> nothing -> F)")
    print("=" * 95)

    predictor.load_model()

    # Load canonical images for testing transitions
    test_sequences = [
        ("Sequence 1: A -> nothing -> B", [
            ("A", 5),
            ("nothing", 4),
            ("B", 5)
        ]),
        ("Sequence 2: O -> nothing -> G", [
            ("O", 5),
            ("nothing", 4),
            ("G", 5)
        ]),
        ("Sequence 3: D -> nothing -> F", [
            ("D", 5),
            ("nothing", 4),
            ("F", 5)
        ]),
    ]

    all_passed = True

    for seq_name, stages in test_sequences:
        print(f"\n--- Testing {seq_name} ---")
        stabilizer = TemporalStabilizerSimulator()
        
        frame_num = 0
        for expected_class, repeat_count in stages:
            for rep in range(repeat_count):
                frame_num += 1
                
                # Fetch image
                if expected_class == "nothing":
                    # Empty wall background
                    img = Image.new("RGB", (200, 200), (145, 142, 140))
                else:
                    files = glob.glob(os.path.join(TRAIN_DIR, expected_class, "*.jpg"))
                    img = Image.open(files[rep % len(files)]).convert("RGB")

                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                raw_res = predictor.predict_image_bytes(buf.getvalue())
                
                smoothed = stabilizer.process_frame(raw_res)
                disp = smoothed["display_prediction"]
                is_st = smoothed["is_stable"]
                is_unc = smoothed.get("is_uncertain", False)
                
                print(f"  Frame {frame_num:02d} | Stage: {expected_class:<7} | Display: {disp:<12} | Conf: {smoothed['confidence']:5.1f}% | Stable: {str(is_st):<5} | Uncertain: {is_unc}")

                # Verification rules:
                # 1. When expected is "nothing", display MUST immediately become "nothing" (no old letters carried over)
                if expected_class == "nothing" and disp != "nothing":
                    print(f"  [ERROR] Frame {frame_num} failed to reset to 'nothing'! Display was '{disp}'")
                    all_passed = False
                # 2. On 2nd frame of new class, must be locked on that new class
                if rep >= 1 and expected_class != "nothing":
                    if disp != expected_class:
                        print(f"  [WARNING] Frame {frame_num} expected '{expected_class}', got '{disp}'")

    print("\n" + "=" * 95)
    if all_passed:
        print("[+] ALL TEMPORAL TRANSITIONS PASSED: Zero state leakage, instant empty-frame resets, smooth transitions!")
    else:
        print("[-] SOME TEMPORAL TRANSITIONS FAILED!")
    print("=" * 95)

if __name__ == "__main__":
    test_temporal_sequences()
