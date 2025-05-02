import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

# Local modules
from .yolo import YOLODetector  # Updated detector in same package

# Resolver stubs (replace with real implementations)
from cv_core.models.facenet import FaceRecognition  # type: ignore
from cv_core.models.car_classifier import CarClassifier  # type: ignore
from cv_core.models.road_sign_ocr import RoadSignOCR  # type: ignore
from cv_core.models.general_object import GeneralObjectClassifier  # type: ignore

# ---------------------------------------------------------------------------
# Configuration – load from our local config.py
# ---------------------------------------------------------------------------
from .config import MODEL_CONFIG, CONFIDENCE_THRESHOLDS, CLASS_NAMES, DEVICE

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_WORKERS = min(8, os.cpu_count() or 1)  # ThreadPool size across images

INFO_LINKS: Dict[int, str] = {
    0: "https://en.wikipedia.org/wiki/Facial_recognition_system",  # Face Recognition
    2: "https://en.wikipedia.org/wiki/Toyota_Camry",               # Car Classifier (example model)
    3: "https://en.wikipedia.org/wiki/Stop_sign",                  # Road Sign OCR
}

# ---------------------------------------------------------------------------
# Utility: JSON-safe type conversion
# ---------------------------------------------------------------------------

def _to_py(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# ModelManager – lazy, **thread‑safe** instantiation of heavy models
# ---------------------------------------------------------------------------

class ModelManager:
    """Singleton‑style registry that lazily loads heavy models.

    A **threading.Lock** is used to guarantee that in multi‑threaded contexts
    (Flask with Gunicorn, FastAPI with Uvicorn workers, etc.) each model is
    constructed exactly once per process.
    """

    _instances: Dict[str, Any] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, name: str):
        # Double‑checked locking for minimal overhead on hot path
        if name in cls._instances:
            return cls._instances[name]

        with cls._lock:
            if name in cls._instances:  # in‑case another thread initialised it
                return cls._instances[name]

            if name == "yolo":
                model = YOLODetector(
                    model_path=MODEL_CONFIG["yolo"],
                    device=DEVICE,
                    class_names=CLASS_NAMES,
                    confidence_thresholds=CONFIDENCE_THRESHOLDS,
                )
            elif name == "face_recognition":
                model = FaceRecognition(MODEL_CONFIG.get("face_recognition"))
            elif name == "car_classifier":
                model = CarClassifier(MODEL_CONFIG.get("car_classifier"))
            elif name == "road_sign_ocr":
                model = RoadSignOCR(MODEL_CONFIG.get("road_sign_ocr"))
            elif name == "general_object":
                model = GeneralObjectClassifier(MODEL_CONFIG.get("general_object"))
            else:
                raise ValueError(f"Unknown model name: {name}")

            cls._instances[name] = model
            logger.info("[ModelManager] Loaded '%s' model", name)
            return model


# ---------------------------------------------------------------------------
# Helper: load image
# ---------------------------------------------------------------------------

def _load_image(path: Union[str, Path]) -> np.ndarray:
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


# ---------------------------------------------------------------------------
# Resolver routing
# ---------------------------------------------------------------------------

def _route_to_resolver(class_id: int, crop: np.ndarray) -> Dict[str, Any]:
    """Route to resolver and enrich with metadata placeholder."""
    meta_stub = {"meta": {"runtime_ms": None, "version": None}}

    start = perf_counter()
    try:
        if class_id == 0:
            model = ModelManager.get("face_recognition")
            out = {"resolver": "face_recognition", "output": model.recognize_faces(crop)}  # type: ignore
        elif class_id == 2:
            model = ModelManager.get("car_classifier")
            out = {"resolver": "car_classifier", "output": model.classify_car(crop)}  # type: ignore
        elif class_id == 3:
            model = ModelManager.get("road_sign_ocr")
            out = {"resolver": "road_sign_ocr", "output": model.extract_text(crop)}  # type: ignore
        elif class_id == 11:
            model = ModelManager.get("road_sign_ocr")
            out = {"resolver": "road_sign_ocr", "output": model.extract_text(crop)}    # type: ignore
        else:
            model = ModelManager.get("general_object")
            out = {"resolver": "general_object", "output": model.classify_object(crop)}  # type: ignore
    except NotImplementedError:
        out = {"note": "resolver not implemented"}
    except Exception as exc:
        logger.error("[Resolve][Error] %s", exc)
        out = {"error": str(exc)}

    runtime_ms = (perf_counter() - start) * 1000
    meta_stub["meta"]["runtime_ms"] = round(runtime_ms, 2)
    out.update(meta_stub)

    # Optional external reference link
    info_link = INFO_LINKS.get(class_id)
    if info_link:
        out["info_link"] = info_link

    return out


# ---------------------------------------------------------------------------
# Image pipeline
# ---------------------------------------------------------------------------

def process_image(path: Union[str, Path], min_conf: float | None = None) -> Dict[str, Any]:
    image_bgr = _load_image(path)
    yolo: YOLODetector = ModelManager.get("yolo")
    detections = yolo.detect(str(path))[0]

    if min_conf is not None:
        detections = [d for d in detections if d["confidence"] >= min_conf]

    logger.info("[Detect] [%s] Detected %d objects.", path, len(detections))

    tasks: List[Tuple[int, Dict[str, Any], np.ndarray]] = []
    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = map(int, (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))
        crop = image_bgr[y1:y2, x1:x2].copy()
        tasks.append((det["class_id"], det, crop))

    results_per_obj: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        fut_map = {exe.submit(_route_to_resolver, cid, crop): meta for cid, meta, crop in tasks}
        for fut in as_completed(fut_map):
            meta = fut_map[fut]
            try:
                meta.update(fut.result())
            except Exception as exc:
                logger.error("[Resolve][Error] %s", exc)
                meta.update({"error": str(exc)})
            results_per_obj.append(_to_py(meta))

    return {"image": str(path), "objects": results_per_obj}


# ---------------------------------------------------------------------------
# Batch pipeline
# ---------------------------------------------------------------------------

def run_pipeline(paths: Sequence[Union[str, Path]], min_conf: float | None = None) -> List[Dict[str, Any]]:
    return [process_image(p, min_conf=min_conf) for p in paths]


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Modular object-detection pipeline")
    parser.add_argument("images", nargs="+", help="Path(s) to image file(s)")
    parser.add_argument("--yolo-weights", dest="weights", help="Custom YOLOv11 weights")
    parser.add_argument("--min-confidence", type=float, help="Global minimum confidence override (0–1)")
    parser.add_argument("--json", action="store_true", help="Print raw JSON to stdout")
    parser.add_argument("--save-json", metavar="FILE", help="Save full results to single JSON file")
    parser.add_argument("--output-dir", metavar="DIR", help="Folder to save one JSON per image")

    args = parser.parse_args()

    if args.weights:
        MODEL_CONFIG["yolo"] = args.weights

    results = run_pipeline(args.images, min_conf=args.min_confidence)

    if args.json or (not args.save_json and not args.output_dir):
        print(json.dumps(results, indent=4, ensure_ascii=False))

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=4, ensure_ascii=False))
        logger.info("[Output] Saved full results to %s", out_path)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for item in results:
            img_name = Path(item["image"]).stem + ".json"
            save_path = out_dir / img_name
            save_path.write_text(json.dumps(item, indent=4, ensure_ascii=False))
            logger.info("[Output] Saved per-image JSON to %s", save_path)
