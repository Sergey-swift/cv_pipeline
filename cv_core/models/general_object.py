from __future__ import annotations

"""general_object.py
~~~~~~~~~~~~~~~~~~~~~
Mock implementation of a *general object* classifier used as a catch‑all
resolver in the modular computer‑vision pipeline. The real model will be
plugged‑in later; for now we return a deterministic placeholder result so
that the rest of the system can be tested end‑to‑end.
"""

from typing import Dict, Any
import logging
import time

import numpy as np

# Optional torch import to honour the *device* argument.  The code does not
# require it, so failure to import will simply disable CUDA detection.
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover – torch not always available
    torch = None  # type: ignore

logger = logging.getLogger(__name__)


class GeneralObjectClassifier:
    """Mock general‑object classifier.

    Parameters
    ----------
    model_config : dict
        Future configuration dictionary (ignored in mock).
    device : str, optional
        Target device ("cuda" or "cpu"). When *None*, the constructor will
        pick **"cuda"** if a GPU with CUDA is available, otherwise **"cpu"**.
    """

    _version: str = "v0.1-mock"

    def __init__(self, model_config: Dict[str, Any] | None = None, device: str | None = None) -> None:
        # Select device – respect caller, else infer from torch.
        if device is not None:
            self.device = device
        else:
            if torch is not None and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

        self.config = model_config or {}
        logger.info("[GeneralObjectClassifier] Initialised on %s", self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_object(self, image: np.ndarray) -> Dict[str, Any]:
        """Classify *image* and return a mock label/probability structure."""
        # Sanity check – ensure we received an ndarray.
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")

        start = time.perf_counter()

        # --- Mock inference – replace with real model forward pass later ----
        # We purposely do *nothing* intensive here – this is just a scaffold.
        mock_label = "mock_label"
        mock_confidence = 0.97
        # -------------------------------------------------------------------

        runtime_ms = (time.perf_counter() - start) * 1000.0

        result: Dict[str, Any] = {
            "object": mock_label,
            "confidence": float(mock_confidence),  # ensure native python
            "meta": {
                "runtime_ms": round(runtime_ms, 3),
                "version": self._version,
            },
        }

        logger.debug("[GeneralObjectClassifier] Mock classification complete")
        return result

    # ------------------------------------------------------------------
    # Stand‑alone smoke test
    # ------------------------------------------------------------------

    if __name__ == "__main__":  # pragma: no cover – manual test only
        import cv2
        import json

        # Create a random 224×224 RGB image.
        dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

        classifier = GeneralObjectClassifier()
        out = classifier.classify_object(dummy_img)

        print("[Test] Classification result:\n" + json.dumps(out, indent=4))
