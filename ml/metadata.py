"""Privacy-preserving metadata summary and auxiliary numeric features."""

from __future__ import annotations

import math
from typing import Final

from PIL import ExifTags, Image


METADATA_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "has_exif",
    "log_file_size",
    "log_pixel_count",
    "log_aspect_ratio",
    "bytes_per_pixel",
    "estimated_compression_ratio",
    "format_jpeg",
    "format_png",
    "format_webp",
    "format_other",
)
METADATA_FEATURE_DIM: Final[int] = len(METADATA_FEATURE_NAMES)


def summarize_metadata(image: Image.Image, file_bytes: bytes) -> dict:
    exif = image.getexif()
    exif_keys = sorted({ExifTags.TAGS.get(key, str(key)) for key in exif.keys()}) if exif else []
    image_format = (image.format or "UNKNOWN").upper()
    channels = 4 if image.mode in {"RGBA", "CMYK"} else 3
    raw_bytes = max(1, image.width * image.height * channels)
    return {
        "has_exif": bool(exif),
        "exif_keys": exif_keys[:20],
        "format": image_format,
        "width": int(image.width),
        "height": int(image.height),
        "mode": image.mode,
        "file_size_bytes": len(file_bytes),
        "estimated_compression_ratio": round(raw_bytes / max(1, len(file_bytes)), 4),
    }


def extract_metadata_features(metadata: dict) -> list[float]:
    """Encode weak provenance/compression context for the learned auxiliary branch."""
    width = max(1, int(metadata["width"]))
    height = max(1, int(metadata["height"]))
    file_size = max(1, int(metadata["file_size_bytes"]))
    pixel_count = width * height
    channels = 4 if metadata.get("mode") in {"RGBA", "CMYK"} else 3
    raw_bytes = pixel_count * channels
    image_format = str(metadata.get("format", "UNKNOWN")).upper()
    return [
        float(bool(metadata.get("has_exif"))),
        math.log1p(file_size),
        math.log1p(pixel_count),
        math.log(width / height),
        file_size / pixel_count,
        raw_bytes / file_size,
        float(image_format in {"JPEG", "JPG"}),
        float(image_format == "PNG"),
        float(image_format == "WEBP"),
        float(image_format not in {"JPEG", "JPG", "PNG", "WEBP"}),
    ]
