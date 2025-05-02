from __future__ import annotations

"""road_sign_ocr.py
~~~~~~~~~~~~~~~~~~~
Final production‑ready OCR / symbol recogniser for cropped road‑sign images.

Two operating modes:
1. **Model mode** – loads a TorchVision/HUB backbone when *weights_path* is
   supplied, patches the classification head to match *num_classes*, and runs
   forward inference (decoder stub for future upgrade).
2. **Mock mode** – if weights are absent, a deterministic colour heuristic
   returns plausible sign labels (*STOP*, *GO*, etc.) to keep the pipeline
   functional during development.

Both modes support batch inputs, FP16 on CUDA, detailed metadata, and optional
logit return.
"""

from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence, Union

import json
import logging
import random

import cv2
import numpy as np
import torch
from torchvision import transforms

try:
    from cv_core.models.model_utils import load_model_weights  # project helper
except ImportError:  # pragma: no cover
    def load_model_weights(path):  # type: ignore
        """Fallback weight loader via ``torch.load``."""
        return torch.load(path, map_location="cpu")

logger = logging.getLogger(__name__)

_VERSION = "v1.0"
_INPUT_SIZE = (320, 320)
_MEAN = [0.5, 0.5, 0.5]
_STD = [0.5, 0.5, 0.5]


class RoadSignOCR:
    """Road‑sign OCR and symbol recognition module."""

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self, model_config: Dict[str, Any] | None = None, device: str | None = None) -> None:
        self.cfg = model_config or {}
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16: bool = bool(self.cfg.get("use_fp16", False) and self.device == "cuda")
        self.return_logits: bool = bool(self.cfg.get("return_logits", False))

        # ---- Load vocabulary if available ----
        self.labels: List[str] = []
        labels_path = self.cfg.get("labels_path")
        if labels_path and Path(labels_path).is_file():
            try:
                with open(labels_path, "r", encoding="utf-8") as f:
                    self.labels = json.load(f)
                if not isinstance(self.labels, list) or not all(isinstance(x, str) for x in self.labels):
                    raise ValueError
            except Exception as exc:
                logger.error("[RoadSignOCR] Failed to load labels: %s", exc)
                self.labels = []
        self.num_classes: int = len(self.labels) if self.labels else 0

        # ---- Model initialisation ----
        self.model: torch.nn.Module | None = None
        self.arch: str | None = None
        weights_path = self.cfg.get("weights_path")
        if weights_path and Path(weights_path).is_file():
            self.arch = self.cfg.get("arch", "resnet18")
            logger.info("[RoadSignOCR] Loading backbone '%s'", self.arch)
            self.model = torch.hub.load("pytorch/vision", self.arch, pretrained=False)
            if self.num_classes:
                self._replace_head(self.model, self.num_classes)
            self.model.load_state_dict(load_model_weights(weights_path), strict=False)
            self.model.to(self.device).eval()
            if self.use_fp16:
                self.model.half()
            logger.info(
                "[RoadSignOCR] Weights loaded (%s classes) • device=%s • fp16=%s",
                self.num_classes or "N/A",
                self.device,
                self.use_fp16,
            )
        else:
            logger.warning("[RoadSignOCR] No weights provided – running in mock mode")

        # ---- Preprocessing transform ----
        self.transform = transforms.Compose(
            [
                transforms.Resize(_INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(_MEAN, _STD),
            ]
        )

    # ------------------------------------------------------------------
    # Head replacement utility (internal)
    # ------------------------------------------------------------------

    @staticmethod
    def _replace_head(model: torch.nn.Module, num_classes: int) -> None:
        """Patch the classification head in *model* to output *num_classes* logits.

        This simplified helper supports typical vision backbones with an
        attribute ``fc``, ``classifier``, or ``head``.  For other architectures
        we will move this logic to a shared utility in the future.
        """
        for attr in ("fc", "classifier", "head"):
            if hasattr(model, attr):
                layer = getattr(model, attr)
                if isinstance(layer, torch.nn.Linear):
                    setattr(model, attr, torch.nn.Linear(layer.in_features, num_classes))
                    return
                if isinstance(layer, torch.nn.Sequential) and isinstance(layer[-1], torch.nn.Linear):
                    in_f = layer[-1].in_features
                    setattr(model, attr, torch.nn.Sequential(*layer[:-1], torch.nn.Linear(in_f, num_classes)))
                    return
        raise RuntimeError("Cannot locate classification head to replace.")

    # ------------------------------------------------------------------
    # Preprocess helpers
    # ------------------------------------------------------------------

    def _preprocess_batch(self, imgs: Sequence[np.ndarray]) -> torch.Tensor:
        tensors: List[torch.Tensor] = []
        for im in imgs:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[RoadSignOCR] Preprocessing image shape=%s dtype=%s", im.shape, im.dtype)
            if im.ndim == 2:
                im = np.stack([im] * 3, axis=-1)
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            t = self.transform(rgb)
            if self.use_fp16:
                t = t.half()
            tensors.append(t)
        return torch.stack(tensors).to(self.device, non_blocking=True)

    # ------------------------------------------------------------------
    # Mock recognition logic
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_recognise(img: np.ndarray) -> tuple[str, float]:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)) | cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
        green = cv2.inRange(hsv, (40, 40, 40), (90, 255, 255))
        if red.sum() / 255 / img.size > 0.05:
            return "STOP", 0.92
        if green.sum() / 255 / img.size > 0.05:
            return "GO", 0.88
        choices = ["Speed Limit 60", "YIELD", "PEDESTRIAN", "No Entry"]
        idx = hash(img.tobytes()) % len(choices)
        return choices[idx], 0.75

    # ------------------------------------------------------------------
    # Decoder placeholder (real model path)
    # ------------------------------------------------------------------

    def _decode(self, logits: torch.Tensor) -> tuple[str, float, int]:
        if not self.labels:
            return "<unk>", 0.5, -1
        probs = torch.softmax(logits, dim=0)
        conf, idx = probs.max(dim=0)
        return self.labels[idx], float(conf), int(idx)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_text(
        self, images: Union[np.ndarray, Sequence[np.ndarray]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        single = isinstance(images, np.ndarray)
        batch = [images] if single else list(images)
        if not all(isinstance(b, np.ndarray) for b in batch):
            raise TypeError("All inputs must be numpy.ndarray")

        start = perf_counter()
        outputs: List[Dict[str, Any]] = []

        # ----- Mock mode -----
        if self.model is None:
            for img in batch:
                text, conf = self._mock_recognise(img)
                meta = {
                    "runtime_ms": round((perf_counter() - start) * 1000 / len(batch), 3),
                    "version": _VERSION,
                }
                outputs.append({"text": text, "confidence": conf, "meta": meta})
            logger.debug("[RoadSignOCR] Mock processed %d sample(s)", len(outputs))
            return outputs[0] if single else outputs

        # ----- Real model path -----
        tensors = self._preprocess_batch(batch)
        with torch.no_grad():
            logits_batch = self.model(tensors)
        runtime = (perf_counter() - start) * 1000.0

        for logits in logits_batch:
            text, conf, idx = self._decode(logits)
            meta = {
                "runtime_ms": round(runtime / len(batch), 3),
                "version": _VERSION,
                "label_index": idx,
                "label_string": text,
                "raw_confidence": conf,
            }
            res: Dict[str, Any] = {"text": text, "confidence": conf, "meta": meta}
            if self.return_logits:
                res["logits"] = logits.tolist()
            outputs.append(res)
        logger.debug("[RoadSignOCR] Model processed %d sample(s)", len(outputs))
        return outputs[0] if single else outputs


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover  – manual testing only
    import argparse
    import json

    parser = argparse.ArgumentParser(description="RoadSignOCR demo")
    parser.add_argument("--img", help="Road‑sign image path", required=False)
    parser.add_argument("--weights")
    parser.add_argument("--labels")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--logits", action="store_true")
    args = parser.parse_args()

    if args.img and Path(args.img).is_file():
        img_np = cv2.imread(args.img)
    else:  # create synthetic green GO sign
        img_np = np.zeros((240, 240, 3), dtype=np.uint8)
        cv2.circle(img_np, (120, 120), 90, (0, 255, 0), -1)

    cfg = {
        "weights_path": args.weights,
        "labels_path": args.labels,
        "use_fp16": args.fp16,
        "return_logits": args.logits,
    }

    ocr = RoadSignOCR(cfg)
    result = ocr.extract_text(img_np)
    print(json.dumps(result, indent=4))
