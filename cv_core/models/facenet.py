from __future__ import annotations

"""face_recognition.py
~~~~~~~~~~~~~~~~~~~~~~
Production-grade identity recogniser for cropped faces, now with enhanced
logging, strict consistency checks, and developer-friendly CLI options.
"""

from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence, Union
import hashlib
import json
import logging
import sys

import cv2
import numpy as np
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)

_VERSION = "v1.2"
_INPUT_SIZE = (160, 160)
_MEAN = [0.5, 0.5, 0.5]
_STD = [0.5, 0.5, 0.5]
_EPS = 1e-5


# ---------------------------------------------------------------------------
# INTERNAL: candidate for shared utils in future refactor 🚀
# ---------------------------------------------------------------------------

def _replace_head(model: torch.nn.Module, out_dim: int) -> None:
    """Replace classification head with Linear(out_dim)."""
    for attr in ("fc", "classifier", "head"):
        if hasattr(model, attr):
            layer = getattr(model, attr)
            if isinstance(layer, torch.nn.Linear):
                setattr(model, attr, torch.nn.Linear(layer.in_features, out_dim))
                return
            if isinstance(layer, torch.nn.Sequential) and isinstance(layer[-1], torch.nn.Linear):
                in_f = layer[-1].in_features
                setattr(model, attr, torch.nn.Sequential(*layer[:-1], torch.nn.Linear(in_f, out_dim)))
                return
    raise RuntimeError("[FaceRecognition] Unsupported backbone head for replacement")


class FaceRecognition:
    """Face identity recognition via cosine-similarity on embeddings."""

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(self, model_config: Dict[str, Any] | None = None, device: str | None = None) -> None:
        self.cfg = model_config or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # --- FP16 guard consistent with other modules ------------------
        # Enabled **only** when caller sets `use_fp16=True` *and* we're on CUDA
        self.fp16: bool = bool(self.cfg.get("use_fp16", False) and self.device == "cuda")

        self.return_embedding: bool = bool(self.cfg.get("return_embedding", False))
        self.min_conf: float = float(self.cfg.get("min_conf", 0.30))
        self.max_batch: int | None = int(self.cfg.get("max_batch", 0)) or None

        # ----------------------------------------------------------------
        # Load optional labels / gallery
        # ----------------------------------------------------------------
        self.labels: List[str] = []
        labels_path = self.cfg.get("labels_path")
        if labels_path and Path(labels_path).is_file():
            with open(labels_path, "r", encoding="utf-8") as f:
                self.labels = json.load(f)
            if not isinstance(self.labels, list) or not all(isinstance(x, str) for x in self.labels):
                raise ValueError("labels_path must point to JSON list[str]")

        self.gallery: torch.Tensor | None = None
        embeddings_path = self.cfg.get("embeddings_path")
        if embeddings_path and Path(embeddings_path).is_file():
            self.gallery = torch.as_tensor(torch.load(embeddings_path, map_location="cpu"))

        if self.labels and self.gallery is not None and len(self.labels) != self.gallery.shape[0]:
            raise ValueError(
                f"labels count ({len(self.labels)}) does not match gallery rows ({self.gallery.shape[0]})"
            )

        # ----------------------------------------------------------------
        # Model (or deterministic mock)
        # ----------------------------------------------------------------
        self.model: torch.nn.Module | None = None
        self.arch: str | None = None
        self.embedding_dim: int = 512  # default mock size

        weights_path = self.cfg.get("weights_path")
        if weights_path and Path(weights_path).is_file():
            self.arch = self.cfg.get("arch", "resnet18")
            logger.info("[FaceRecognition] Loading backbone '%s'", self.arch)
            self.model = torch.hub.load("pytorch/vision", self.arch, pretrained=False)
            # Determine embedding dim automatically from backbone head
            if hasattr(self.model, "fc") and isinstance(self.model.fc, torch.nn.Linear):
                self.embedding_dim = self.model.fc.in_features
            _replace_head(self.model, self.embedding_dim)

            self.model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
            self.model.to(self.device).eval()
            if self.fp16:
                self.model.half()

            n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(
                "[FaceRecognition] Model ready • params=%d • dim=%d • device=%s • fp16=%s",
                n_params,
                self.embedding_dim,
                self.device,
                self.fp16,
            )
        else:
            logger.warning("[FaceRecognition] No weights supplied – using deterministic mock mode")

        # Embedding-dim sanity check
        if self.gallery is not None and self.gallery.shape[1] != self.embedding_dim:
            logger.warning(
                "[FaceRecognition] Embedding dim mismatch (model=%d, gallery=%d). Matching may be unreliable.",
                self.embedding_dim,
                self.gallery.shape[1],
            )

        # ----------------------------------------------------------------
        # Pre-processing transform
        # ----------------------------------------------------------------
        self.transform = transforms.Compose(
            [
                transforms.Resize(_INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(_MEAN, _STD),
            ]
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _preprocess_batch(self, imgs: Sequence[np.ndarray]) -> torch.Tensor:
        tensors = []
        for im in imgs:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[FaceRecognition] preprocessing image shape=%s", im.shape)
            if im.ndim == 2:
                im = np.stack([im] * 3, axis=-1)
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            t = self.transform(rgb)
            if self.fp16:
                t = t.half()
            tensors.append(t)
        return torch.stack(tensors).to(self.device)

    def _mock_embed(self, img: np.ndarray) -> torch.Tensor:
        sha = hashlib.sha256(img.tobytes()).digest()
        arr = np.frombuffer(sha, dtype=np.uint8).astype(np.float32)
        arr = np.pad(arr, (0, self.embedding_dim - len(arr)), "wrap")[: self.embedding_dim]
        arr /= np.linalg.norm(arr) + _EPS
        return torch.from_numpy(arr)

    def _embed_batch(self, imgs: Sequence[np.ndarray]) -> torch.Tensor:
        if self.model is None:
            return torch.stack([self._mock_embed(im) for im in imgs])
        inp = self._preprocess_batch(imgs)
        with torch.no_grad():
            out = self.model(inp).float().cpu()
        return torch.nn.functional.normalize(out, dim=1, eps=_EPS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def recognize_faces(
        self,
        faces: Union[np.ndarray, Sequence[np.ndarray]],
        *,
        min_confidence: float | None = None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Identify faces with optional **min_confidence** override."""
        single = isinstance(faces, np.ndarray)
        batch = [faces] if single else list(faces)
        if self.max_batch and len(batch) > self.max_batch:
            raise ValueError("Batch size exceeds max_batch")
        if not all(isinstance(x, np.ndarray) for x in batch):
            raise TypeError("All inputs must be numpy.ndarray")

        min_conf = float(min_confidence) if min_confidence is not None else self.min_conf

        t0 = perf_counter()
        embeddings = self._embed_batch(batch)
        runtime = (perf_counter() - t0) * 1000.0

        results = []
        for emb in embeddings:
            identity, conf, idx = "Unknown", 0.0, -1
            if self.gallery is not None:
                sims = torch.matmul(self.gallery, emb) / (
                    self.gallery.norm(dim=1) * emb.norm() + _EPS
                )
                conf_t, idx_t = sims.max(dim=0)
                conf, idx = float(conf_t), int(idx_t)
                if conf >= min_conf:
                    identity = self.labels[idx] if 0 <= idx < len(self.labels) else f"ID_{idx}"
            else:
                # Deterministic mock identity
                identity = f"Person_{hashlib.sha256(emb.numpy().tobytes()).hexdigest()[:8]}"
                conf = 0.50

            meta = {
                "runtime_ms": round(runtime / len(batch), 3),
                "version": _VERSION,
                "embedding_dim": self.embedding_dim,
                "identity_index": idx,
            }
            res: Dict[str, Any] = {"identity": identity, "confidence": conf, "meta": meta}
            if self.return_embedding:
                res["embedding"] = emb.tolist()
            results.append(res)

        logger.debug("[FaceRecognition] processed %d face(s)", len(results))
        return results[0] if single else results


# ---------------------------------------------------------------------------
# CLI utility (manual smoke-test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="FaceRecognition quick demo")
    parser.add_argument("--img", help="Path to face crop (BGR)")
    parser.add_argument("--weights")
    parser.add_argument("--labels")
    parser.add_argument("--gallery")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--emb", action="store_true", help="Return embedding in output")
    parser.add_argument("--min-conf", type=float, help="Override min confidence threshold")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--dry-run", action="store_true", help="Initialise model then exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    cfg = {
        "weights_path": args.weights,
        "labels_path": args.labels,
        "embeddings_path": args.gallery,
        "use_fp16": args.fp16,
        "return_embedding": args.emb,
    }

    recog = FaceRecognition(cfg)

    if args.dry_run:
        logger.info("[FaceRecognition] Dry-run successful – model initialised.")
        sys.exit(0)

    if args.img and Path(args.img).is_file():
        img_np = cv2.imread(args.img)
    else:
        img_np = np.random.randint(0, 256, (160, 160, 3), dtype=np.uint8)
        logger.warning("[FaceRecognition] Using random test image as --img not provided or invalid")

    output = recog.recognize_faces(img_np, min_confidence=args.min_conf)
    print(json.dumps(output, indent=4))
