from __future__ import annotations

from PIL import ExifTags, Image


def _decode_exif_keys(exif: dict) -> list[str]:
    names: list[str] = []
    for key in exif.keys():
        names.append(ExifTags.TAGS.get(key, str(key)))
    return sorted(set(names))


def check_metadata(image: Image.Image, file_bytes: bytes) -> dict:
    """
    Extract a privacy-preserving image metadata summary.

    Raw image bytes and full EXIF values are intentionally not returned or logged.
    """
    exif = image.getexif()
    exif_keys = _decode_exif_keys(exif) if exif else []

    return {
        "has_exif": bool(exif),
        "exif_keys": exif_keys[:20],
        "format": image.format or "UNKNOWN",
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "file_size_bytes": len(file_bytes),
    }
