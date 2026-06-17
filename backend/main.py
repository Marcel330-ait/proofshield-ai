from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from detector import detect_ai_generated
from metadata_checker import check_metadata
from risk_report import generate_report

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

app = FastAPI(
    title="ProofShield AI",
    description="Local-first AI-generated image risk signal API.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ProofShield AI"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Please upload a JPG, PNG, or WEBP image.",
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File is too large. Maximum upload size is 10MB.",
        )

    try:
        with Image.open(BytesIO(file_bytes)) as image:
            image.verify()
        with Image.open(BytesIO(file_bytes)) as image:
            image.load()
            metadata = check_metadata(image, file_bytes)
            ai_score = detect_ai_generated(image, metadata)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be safely opened as an image.",
        ) from exc

    report = generate_report(ai_score, metadata)
    return {
        "ai_probability": ai_score,
        "risk_level": report["risk_level"],
        "conclusion": report["conclusion"],
        "signals": report["signals"],
        "recommendations": report["recommendations"],
        "localized": report["localized"],
        "metadata": metadata,
        "disclaimer": "This tool provides an AI-generated risk signal only. It does not prove whether an image is real or fake.",
        "disclaimer_localized": {
            "en": "This tool provides an AI-generated risk signal only. It does not prove whether an image is real or fake.",
            "zh": "本工具仅提供 AI 生成风险信号，并不能证明图片是真实或伪造的。",
        },
    }
