# ────────────────────────────────────────────────────────────────────────────────
#  Dockerfile – production-ready container for the “cv_pipeline” Flask project
#  ---------------------------------------------------------------------------
#  • Python 3.11-slim for minimal image size
#  • Installs only the OS libraries required by OpenCV & python-magic
#  • Uses Gunicorn so the container is production-safe out of the box
#  • WORKDIR is the repo root ( /srv/cv_pipeline ) – Flask finds templates/static
# ────────────────────────────────────────────────────────────────────────────────

# ---------- 1️⃣  Base image -----------------------------------------------------
FROM python:3.11-slim AS runtime

# ---------- 2️⃣  Runtime env vars ---------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Tweaks for PyTorch/OpenCV inside containers
    OMP_NUM_THREADS=1 \
    # Gunicorn default settings (can be overridden at `docker run`)
    GUNICORN_CMD_ARGS="--workers=3 --timeout=120 --graceful-timeout=30"

# ---------- 3️⃣  OS-level deps (minimal) ---------------------------------------
#  • libmagic1   – MIME sniffing in app.py
#  • libglib2.0-0, libsm6, libxext6 – OpenCV headless at runtime
#  • gcc         – wheels that need tiny C extensions (e.g. python-magic)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libmagic1 libglib2.0-0 libsm6 libxext6 libgl1 gcc && \
    rm -rf /var/lib/apt/lists/*

# ---------- 4️⃣  Project setup --------------------------------------------------
# Workdir = repo root so relative paths in code stay unchanged
WORKDIR /srv/cv_pipeline

# Copy dependency list first – allows Docker-layer caching
COPY requirements.txt .

# Upgrade pip then install deps without cache to keep image small
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the source code (observes .dockerignore)
COPY . .

# ---------- 5️⃣  Network / ports ------------------------------------------------
EXPOSE 5000

# ---------- 6️⃣  Entrypoint -----------------------------------------------------
#  Gunicorn entry: “app.app:app”  →  package.folder:file (Flask ‘app’ instance)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.app:app"]
