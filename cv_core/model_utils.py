from __future__ import annotations

"""model_utils.py
~~~~~~~~~~~~~~~~~
Shared helper utilities for model wrappers inside the modular CV pipeline.
Importing this module is *side‑effect‑free* — it performs no I/O at import
and has no project‑specific dependencies.
"""

from pathlib import Path
from typing import Any, List, Mapping, Sequence, Union, overload
import json
import logging
import os

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typing aliases
# ---------------------------------------------------------------------------

PathLike = Union[str, os.PathLike]

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _ensure_file(path: Path, desc: str) -> None:
    """Raise *FileNotFoundError* with logged message if *path* does not exist."""
    if not path.is_file():
        msg = f"{desc} not found: {path}"
        logger.error("[model_utils] %s", msg)
        raise FileNotFoundError(msg)

# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device(*, force_cpu: bool = False) -> str:
    """Return the best‑available device as a string.

    Parameters
    ----------
    force_cpu : bool, optional
        If *True*, always return ``"cpu"``.

    Returns
    -------
    str
        ``"cuda"`` when available (and *force_cpu* is *False*); otherwise
        ``"cpu"``.
    """
    return "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"

# ---------------------------------------------------------------------------
# Weight I/O
# ---------------------------------------------------------------------------

@overload
def load_model_weights(weights_path: PathLike) -> Mapping[str, Any]:
    ...  # pragma: no cover

@overload
def load_model_weights(weights_path: PathLike, *, map_location: str | torch.device) -> Mapping[str, Any]:
    ...  # pragma: no cover

def load_model_weights(
    weights_path: PathLike,
    *,
    map_location: str | torch.device = "cpu",
) -> Mapping[str, Any]:
    """Load a PyTorch *state_dict* from **weights_path**.

    Parameters
    ----------
    weights_path : PathLike
        File path to ``.pt`` / ``.pth`` containing a serialized *state_dict*.
    map_location : str | torch.device, optional
        Device mapping for tensors (defaults to ``"cpu"``).

    Returns
    -------
    Mapping[str, Any]
        Loaded state dictionary.

    Raises
    ------
    FileNotFoundError
        If *weights_path* does not exist.
    RuntimeError
        If deserialization fails.
    """
    path = Path(weights_path)
    _ensure_file(path, "Weights file")
    try:
        state = torch.load(path, map_location=map_location)
        logger.info("[model_utils] Weights loaded from %s (keys=%d)", path, len(state))
        return state
    except Exception as exc:  # pragma: no cover
        logger.exception("[model_utils] Failed to load weights from %s: %s", path, exc)
        raise


def save_model_weights(model: torch.nn.Module, path: PathLike) -> None:
    """Save **model** parameters to disk.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose ``state_dict`` is to be written.
    path : PathLike
        Destination file path. Parent directories are created automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info("[model_utils] Weights saved to %s", path)

# ---------------------------------------------------------------------------
# Model surgery helpers
# ---------------------------------------------------------------------------

def replace_head(model: torch.nn.Module, out_dim: int) -> None:
    """Replace classifier/projection head *in‑place*.

    The function searches for attributes ``fc``, ``classifier``, or ``head``.
    If the attribute is a ``Linear`` layer, it is swapped with a new
    ``Linear(in_features, out_dim)``. If the attribute is a ``Sequential``
    whose last block is ``Linear``, only the final block is replaced.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be modified.
    out_dim : int
        Desired output dimension.

    Raises
    ------
    RuntimeError
        If a supported head cannot be found.
    """
    for attr in ("fc", "classifier", "head"):
        if not hasattr(model, attr):
            continue
        block = getattr(model, attr)
        # Single Linear head
        if isinstance(block, torch.nn.Linear):
            logger.debug(
                "[model_utils] Replacing %s Linear head: %d → %d", attr, block.out_features, out_dim
            )
            setattr(model, attr, torch.nn.Linear(block.in_features, out_dim))
            return
        # Sequential ending in Linear
        if isinstance(block, torch.nn.Sequential) and isinstance(block[-1], torch.nn.Linear):
            in_f = block[-1].in_features
            new_seq = torch.nn.Sequential(*block[:-1], torch.nn.Linear(in_f, out_dim))
            setattr(model, attr, new_seq)
            logger.debug("[model_utils] Replaced %s Sequential head (→ %d)", attr, out_dim)
            return
    raise RuntimeError("[model_utils] Unsupported backbone — head replacement failed")

# ---------------------------------------------------------------------------
# Label loader
# ---------------------------------------------------------------------------

def load_class_labels(label_path: PathLike) -> List[str]:
    """Load class labels from a JSON **list[str]** file.

    Parameters
    ----------
    label_path : PathLike
        Path to JSON file.

    Returns
    -------
    list[str]
        Loaded labels, or ``["Unknown"]`` on failure.
    """
    path = Path(label_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
            raise ValueError("JSON must be list[str]")
        return data
    except Exception as exc:  # pragma: no cover
        logger.warning("[model_utils] Failed to load labels from %s: %s", path, exc)
        return ["Unknown"]

# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def safe_json_write(path: PathLike, data: Any, *, indent: int = 4) -> None:
    """Safely write *data* to *path* formatted as JSON.

    Parent directories are created automatically. Errors are re‑raised after
    logging.
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=indent, ensure_ascii=False))
        logger.info("[model_utils] JSON written → %s", path)
    except Exception as exc:  # pragma: no cover
        logger.error("[model_utils] Could not write JSON to %s: %s", path, exc)
        raise


def print_model_summary(
    model: torch.nn.Module, *, logger_level: int = logging.INFO, return_dict: bool = False
) -> dict[str, Any] | None:
    """Log (and optionally return) a concise model summary.

    Parameters
    ----------
    model : torch.nn.Module
        The model to summarise.
    logger_level : int, optional
        Logging level for the summary line (default: ``logging.INFO``).
    return_dict : bool, optional
        If *True*, also return a dictionary with summary stats.

    Returns
    -------
    dict[str, Any] | None
        Summary dictionary when *return_dict* is *True*, otherwise *None*.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    bytes_total = sum(p.element_size() * p.numel() for p in model.parameters())
    size_mb = bytes_total / (1024 ** 2)

    logger.log(
        logger_level,
        "[model_utils] Summary: trainable=%d • total=%d • size=%.2f MB",
        trainable,
        total,
        size_mb,
    )

    if return_dict:
        return {"trainable": trainable, "total": total, "size_mb": size_mb}
    return None

# ---------------------------------------------------------------------------
# Self‑test (manual)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
    # Device helper test
    print("Selected device:", get_device())
    # Label loader test (expected warning)
    print("Labels:", load_class_labels("nonexistent.json"))
