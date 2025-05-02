from __future__ import annotations

"""car_classifier.py
~~~~~~~~~~~~~~~~~~~~
High-quality car make / model classifier designed for production pipelines.
The class dynamically adapts to different TorchVision/HUB backbones, supports
half-precision on CUDA, optional logits return, rich per-sample metadata, and
batch-efficient inference.
"""

from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence, Union
import json
import logging

import numpy as np
import torch
from torchvision import transforms

try:
    from cv_core.models.model_utils import load_model_weights          # project helper
except ImportError:                                            # pragma: no cover
    def load_model_weights(path: Union[str, Path]):            # type: ignore
        """Fallback weight loader using ``torch.load``."""
        return torch.load(path, map_location="cpu")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
_INPUT_SIZE = (224, 224)
_VERSION = "v0.2"

# ---------------------------------------------------------------------------
# Utility – generic head patcher
# ---------------------------------------------------------------------------
def _replace_head(model: torch.nn.Module, num_classes: int) -> None:
    """Swap the classification head with ``Linear(num_classes)``."""
    for attr in ("fc", "classifier", "head"):
        if hasattr(model, attr):
            block = getattr(model, attr)
            # Single Linear head
            if isinstance(block, torch.nn.Linear):
                setattr(model, attr, torch.nn.Linear(block.in_features, num_classes))
                return
            # Sequential ending in Linear
            if isinstance(block, torch.nn.Sequential) and isinstance(block[-1], torch.nn.Linear):
                in_f = block[-1].in_features
                new_seq = list(block[:-1]) + [torch.nn.Linear(in_f, num_classes)]
                setattr(model, attr, torch.nn.Sequential(*new_seq))
                return
    # Vision-Transformer style heads (recursive)
    if hasattr(model, "heads") and isinstance(model.heads, torch.nn.Module):
        try:
            _replace_head(model.heads, num_classes)
            return
        except Exception:
            pass
    raise RuntimeError("Unable to locate classification head to replace.")

# ---------------------------------------------------------------------------
# CarClassifier
# ---------------------------------------------------------------------------
class CarClassifier:
    """Recognise car make / model from cropped RGB images."""

    def __init__(self, model_config: Dict[str, Any], device: str | None = None) -> None:
        self.cfg = model_config or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # -------- FP16 guard (consistent) --------
        self.fp16: bool = bool(self.cfg.get("use_fp16", False) and self.device == "cuda")

        self.return_logits = bool(self.cfg.get("return_logits", False))
        self.min_conf = float(self.cfg.get("min_conf", 0.20))
        self.max_batch = int(self.cfg.get("max_batch", 0)) or None

        # -------- Labels --------
        labels_path = self.cfg.get("labels_path")
        if not labels_path or not Path(labels_path).is_file():
            raise FileNotFoundError("labels_path must point to valid JSON file")
        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels: List[str] = json.load(f)
        if not isinstance(self.labels, list) or not all(isinstance(l, str) for l in self.labels):
            raise ValueError("labels file must be a JSON list[str]")
        self.num_classes = len(self.labels)

        # -------- Backbone --------
        arch = self.cfg.get("arch", "resnext101_32x8d_wsl")
        logger.info("[CarClassifier] Loading backbone '%s'", arch)
        self.model = torch.hub.load("pytorch/vision", arch, pretrained=False)
        _replace_head(self.model, self.num_classes)

        weights_path = self.cfg.get("weights_path")
        if not weights_path or not Path(weights_path).is_file():
            raise FileNotFoundError("weights_path must point to existing model weights")
        state_dict = load_model_weights(weights_path)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()

        if self.fp16:
            self.model.half()
            logger.info("[CarClassifier] FP16 inference enabled")

        # -------- Pre-processing --------
        self.transform = transforms.Compose(
            [
                transforms.Resize(_INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(_MEAN, _STD),
            ]
        )

        logger.info(
            "[CarClassifier] Ready – %d classes • device=%s • fp16=%s",
            self.num_classes,
            self.device,
            self.fp16,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _preprocess_batch(self, imgs: Sequence[np.ndarray]) -> torch.Tensor:
        tensors: List[torch.Tensor] = []
        for img in imgs:
            if not isinstance(img, np.ndarray):
                raise TypeError("Each image must be np.ndarray")
            if img.ndim == 2:                          # gray → 3-ch
                img = np.stack([img] * 3, axis=-1)
            img_rgb = img[..., ::-1]                   # BGR → RGB
            pil = transforms.ToPILImage()(img_rgb.astype(np.uint8))
            t = self.transform(pil)
            if self.fp16:
                t = t.half()
            tensors.append(t)
        return torch.stack(tensors).to(self.device, non_blocking=True)

    def _decode_label(self, idx: int) -> tuple[str, str]:
        label = self.labels[idx]
        parts = label.split(" ", 1)
        return (parts[0], parts[1]) if len(parts) == 2 else (label, "Unknown")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify_car(
        self, car_images: Union[np.ndarray, Sequence[np.ndarray]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        # Normalise input to list
        if isinstance(car_images, np.ndarray):
            batch = [car_images]
            single_input = True
        elif isinstance(car_images, (list, tuple)) and all(isinstance(x, np.ndarray) for x in car_images):
            batch = list(car_images)
            single_input = False
        else:
            raise TypeError("car_images must be ndarray or list of ndarray")

        if self.max_batch and len(batch) > self.max_batch:
            raise ValueError(f"Batch size {len(batch)} exceeds max_batch {self.max_batch}")

        inputs = self._preprocess_batch(batch)

        start = perf_counter()
        with torch.no_grad():
            logits: torch.Tensor = self.model(inputs)
            probs = torch.nn.functional.softmax(logits, dim=1)
        runtime = (perf_counter() - start) * 1000.0

        confs, preds = probs.max(dim=1)

        results: List[Dict[str, Any]] = []
        for i, (conf, pred_idx) in enumerate(zip(confs.tolist(), preds.tolist())):
            make, model = ("Unknown", "Unknown")
            if conf >= self.min_conf:
                make, model = self._decode_label(pred_idx)
            meta = {
                "runtime_ms": round(runtime / len(batch), 3),
                "version": _VERSION,
                "label_index": pred_idx,
                "label_string": self.labels[pred_idx],
                "raw_confidence": float(conf),
            }
            out: Dict[str, Any] = {
                "make": make,
                "model": model,
                "confidence": float(conf),
                "meta": meta,
            }
            if self.return_logits:
                out["logits"] = logits[i].tolist()
            results.append(out)

        logger.debug("[CarClassifier] Classified %d car(s)", len(results))
        return results[0] if single_input else results

# ---------------------------------------------------------------------------
# Stand-alone sanity test
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import cv2
    import argparse

    parser = argparse.ArgumentParser(description="CarClassifier quick test")
    parser.add_argument("--img", required=False, help="Path to car image")
    parser.add_argument("--weights", required=True, help="Weights .pt file")
    parser.add_argument("--labels", required=True, help="Labels .json list")
    args = parser.parse_args()

    img_arr = (
        cv2.imread(args.img) if args.img else np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    )

    cfg = {
        "weights_path": args.weights,
        "labels_path": args.labels,
        "use_fp16": False,
        "return_logits": True,
    }

    classifier = CarClassifier(cfg)
    import json as _j
    print(_j.dumps(classifier.classify_car(img_arr), indent=4))
