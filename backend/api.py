"""FastAPI serving layer for a validated local ProofShield AI v2 checkpoint."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import anyio
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.inference import LocalInferenceEngine, ModelNotReadyError  # noqa: E402


ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "model.pt"

app = FastAPI(
    title="ProofShield AI v2",
    version="2.0.0",
    description="Local multimodal AI-generated image risk estimation. Requires a validated local checkpoint.",
)
allowed_origins = {
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://marcel330-ait.github.io",
}
allowed_origins.update(
    origin.strip()
    for origin in os.environ.get("PROOFSHIELD_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def checkpoint_path() -> Path:
    return Path(os.environ.get("PROOFSHIELD_CHECKPOINT", str(DEFAULT_CHECKPOINT))).expanduser()


@lru_cache(maxsize=1)
def get_engine() -> LocalInferenceEngine:
    return LocalInferenceEngine(checkpoint_path())


@app.get("/health")
def health() -> dict:
    path = checkpoint_path()
    return {
        "status": "ready" if path.is_file() else "model_not_ready",
        "service": "ProofShield AI v2",
        "checkpoint": str(path),
        "inference_mode": "local_only",
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, "Unsupported file type. Upload JPG, PNG, or WEBP.")
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(413, "File is too large. Maximum upload size is 25MB.")
    try:
        with Image.open(__import__("io").BytesIO(file_bytes)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, "The uploaded file could not be safely opened as an image.") from exc
    try:
        engine = get_engine()
    except ModelNotReadyError as exc:
        raise HTTPException(503, f"Model is not ready. Train or configure a validated checkpoint first. {exc}") from exc
    return await anyio.to_thread.run_sync(engine.analyze_bytes, file_bytes)
