"""
config.py
~~~~~~~~~
Central configuration for the modular CV pipeline and Flask demo.

* Paths / flags can now be **over-ridden via environment variables** so the
  same image can run unchanged on local dev-boxes, CI and Docker.

    ── Supported ENV keys ────────────────────────────────────────────────
      • CV_ROOT_DIR      – project root (defaults to this file’s parent)
      • WEIGHT_DIR       – override default `models/weights`
      • LABEL_DIR        – override default `models/labels`
      • YOLO_WEIGHTS     – custom detector weights (.pt)
      • USE_FP16         – “1|true|yes” → enable half-precision inference
      • DEVICE           – force “cpu” / “cuda” / “mps” (otherwise auto)
    ───────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bool_env(var: str, default: bool = False) -> bool:
    """Convert truth-y env strings to bool."""
    return os.getenv(var, str(int(default))).strip().lower() in {"1", "true", "yes"}


def _env_path(var: str, default: Path) -> Path:
    """Return *default* unless env-var points to a valid path string."""
    return Path(os.getenv(var, str(default))).expanduser().resolve()


# ---------------------------------------------------------------------------
# Repo-relative directories (all overridable)
# ---------------------------------------------------------------------------

ROOT_DIR: Path = _env_path("CV_ROOT_DIR", Path(__file__).resolve().parent)
WEIGHT_DIR: Path = _env_path("WEIGHT_DIR", ROOT_DIR / "models" / "weights")
LABEL_DIR: Path = _env_path("LABEL_DIR", ROOT_DIR / "models" / "labels")

# ---------------------------------------------------------------------------
# Device helper (optional torch import keeps config lightweight)
# ---------------------------------------------------------------------------

try:
    import torch  # noqa: WPS433

    _FORCED_DEVICE = os.getenv("DEVICE")
    DEVICE = _FORCED_DEVICE if _FORCED_DEVICE else ("cuda" if torch.cuda.is_available() else "cpu")
except Exception:                       # pragma: no cover
    DEVICE = os.getenv("DEVICE", "cpu")  # fall back to env or plain CPU

USE_FP16_GLOBAL: bool = _bool_env("USE_FP16", default=False)

# ---------------------------------------------------------------------------
# Model configuration dictionary (paths first, then per-resolver flags)
# ---------------------------------------------------------------------------

MODEL_CONFIG: Dict[str, Union[Dict, str, None]] = {
    # -------------------- Detector --------------------
    "yolo": os.getenv("YOLO_WEIGHTS", str(WEIGHT_DIR / "yolo11n.pt")),

    # -------------------- Resolvers -------------------
    "face_recognition": {
        "weights_path": str(WEIGHT_DIR / "face_recognition.pt"),
        "labels_path": str(LABEL_DIR / "face_labels.json"),
        "embeddings_path": str(WEIGHT_DIR / "face_gallery.pt"),
        "use_fp16": USE_FP16_GLOBAL,
    },
    "car_classifier": {
        "weights_path": str(WEIGHT_DIR / "car_classifier.pt"),
        "labels_path": str(LABEL_DIR / "car_labels.json"),
        "arch": os.getenv("CAR_BACKBONE", "resnext101_32x8d_wsl"),
        "use_fp16": USE_FP16_GLOBAL,
    },
    "road_sign_ocr": {
        # empty string → stays in mock-mode
        "weights_path": os.getenv("OCR_WEIGHTS", str(WEIGHT_DIR / "road_sign_ocr.pt")),
        "labels_path": str(LABEL_DIR / "road_sign_labels.json"),
        "use_fp16": USE_FP16_GLOBAL,
    },
    # Mock resolver – left explicit for clarity
    "general_object": None,
}

# ---------------------------------------------------------------------------
# YOLO per-class confidence thresholds
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLDS: Dict[int, float] = {
    0: float(os.getenv("THRESH_PERSON", 0.25)),
    2: float(os.getenv("THRESH_CAR", 0.30)),
    3: float(os.getenv("THRESH_SIGN", 0.30)),
}

# ---------------------------------------------------------------------------
# Optional class names list (custom datasets only)
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    # extend as needed
]

# ---------------------------------------------------------------------------
# Sanity-check helper (executed when running `python config.py`)
# ---------------------------------------------------------------------------

def _warn_missing_paths() -> None:
    """Warn if any configured weight/label file does not exist."""
    def _check(path_str: str, kind: str) -> None:
        p = Path(path_str)
        if path_str and not p.is_file():
            logger.warning("[config] %s file not found: %s", kind, p)

    for key, cfg in MODEL_CONFIG.items():
        if isinstance(cfg, str):
            _check(cfg, f"{key} weights")
        elif isinstance(cfg, dict):
            for k in ("weights_path", "labels_path", "embeddings_path"):
                if k in cfg:
                    _check(cfg[k], f"{key}:{k}")


# ---------------------------------------------------------------------------
# Stand-alone diagnostic
# ---------------------------------------------------------------------------

if __name__ == "__main__":              # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    print("DEVICE:", DEVICE, "• FP16:", USE_FP16_GLOBAL)
    print("YOLO weights path:", MODEL_CONFIG["yolo"])
    print("Confidence thresholds:", json.dumps(CONFIDENCE_THRESHOLDS, indent=2))
    _warn_missing_paths()
