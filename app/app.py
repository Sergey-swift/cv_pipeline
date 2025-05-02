"""
Flask wrapper for the modular CV pipeline.

Minor refinements:
• Pass class-names into bounding-box renderer.
• Lightweight warm-up to avoid cold-start latency.
• Tiny disk-bloat guard (keep max 200 files in static folders).
• **NEW** MIME sniffing + 413 handler for extra upload safety.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List

import cv2
from flask import (
    Flask,
    abort,
    render_template,
    request,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Local CV pipeline
# ---------------------------------------------------------------------------
from cv_core.main import run_pipeline, ModelManager  # type: ignore
from cv_core.yolo import YOLODetector  # draw helper

try:                                # optional, for class-label overlay
    from config import CLASS_NAMES  # type: ignore
except Exception:                   # pragma: no cover
    CLASS_NAMES: list[str] | None = None
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# MIME / type-sniffing
# ---------------------------------------------------------------------------
try:
    import magic  # python-magic -- system libmagic required
    _sniff = lambda p: magic.from_file(str(p), mime=True)            # noqa: E731
except Exception:                             # fallback – cheap Pillow probe
    from PIL import Image  # type: ignore

    def _sniff(path: Path) -> str:            # noqa: D401
        try:
            with Image.open(path) as im:
                return Image.MIME.get(im.format, "application/octet-stream")
        except Exception:
            return "application/octet-stream"

ALLOWED_EXT   = {".jpg", ".jpeg", ".png", ".bmp"}
ALLOWED_MIME  = {"image/jpeg", "image/png", "image/bmp"}
_MAX_FILES_PER_DIR = 200      # simple disk-bloat guard
_MAX_UPLOAD_MB     = 8

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("web")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_ROOT   = Path(__file__).parent.resolve()
STATIC_DIR = APP_ROOT / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
RESULT_DIR = STATIC_DIR / "results"
for d in (UPLOAD_DIR, RESULT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_allowed_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT


def _timestamp_fname(fname: str, prefix: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    stem, ext = Path(fname).stem, Path(fname).suffix
    return f"{prefix}_{stem}_{ts}{ext}"


def _cleanup_dir(directory: Path) -> None:
    files = sorted(directory.iterdir(), key=lambda p: p.stat().st_mtime)
    while len(files) > _MAX_FILES_PER_DIR:
        files[0].unlink(missing_ok=True)
        files.pop(0)


def _annotate(src_path: Path, detections: List[Dict[str, Any]]) -> Path:
    """Draw boxes on *src_path* and save under RESULT_DIR."""
    img = cv2.imread(str(src_path))
    if img is None:
        raise ValueError("Cannot read uploaded image")

    annotated = YOLODetector.draw_boxes(img, detections, class_names=CLASS_NAMES)
    out_path = RESULT_DIR / _timestamp_fname(src_path.name, "annotated")
    cv2.imwrite(str(out_path), annotated)
    _cleanup_dir(RESULT_DIR)
    return out_path


def _rel_static_url(path: Path) -> str:
    return url_for("static", filename=str(path.relative_to(STATIC_DIR)))


def _warm_up_models() -> None:
    """Pre-load heavy models once at startup to cut first-request latency."""
    logger.info("[WarmUp] Loading detector and resolvers…")
    ModelManager.get("yolo")            # detector
    ModelManager.get("general_object")  # common resolver
    logger.info("[WarmUp] Completed.")


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = _MAX_UPLOAD_MB * 1024 * 1024  # 8 MB

# kick off warm-up in background so app starts immediately
Thread(target=_warm_up_models, daemon=True).start()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("image")
    if not file or file.filename == "":
        abort(400, "No file selected.")
    if not _is_allowed_extension(file.filename):
        abort(415, "Unsupported file type (extension).")

    # ---- persist upload safely ------------------------------------------------
    fname = secure_filename(file.filename)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(fname).suffix)
    file.save(tmp.name)
    tmp_path = Path(tmp.name)

    # ---- MIME sniffing --------------------------------------------------------
    mime = _sniff(tmp_path)
    if mime not in ALLOWED_MIME:
        tmp_path.unlink(missing_ok=True)
        abort(415, f"Unsupported MIME type: {mime}")

    upload_path = UPLOAD_DIR / _timestamp_fname(fname, "upload")
    shutil.move(tmp_path, upload_path)
    _cleanup_dir(UPLOAD_DIR)

    # ---- run pipeline ---------------------------------------------------------
    logger.info("[Web] Processing %s", upload_path.name)
    result = run_pipeline([upload_path])[0]
    detections = result.get("objects", [])

    # ---- annotate (if any) ----------------------------------------------------
    img_path = _annotate(upload_path, detections) if detections else upload_path
    img_url = _rel_static_url(img_path)

    return render_template(
        "results.html",
        image_url=img_url,
        objects=detections,
        raw_json=json.dumps(result, indent=4, ensure_ascii=False),
    )

# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(RequestEntityTooLarge)        # 413
def _too_large(err):                            # type: ignore
    msg = f"File exceeds {_MAX_UPLOAD_MB} MB limit."
    return render_template("error.html", message=msg), 413


@app.errorhandler(400)
@app.errorhandler(415)
def _bad_request(err):                          # type: ignore
    return render_template("error.html", message=str(err)), err.code


@app.errorhandler(500)
def _server_error(err):                         # type: ignore
    logger.exception(err)
    return render_template("error.html", message="Internal server error."), 500


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
