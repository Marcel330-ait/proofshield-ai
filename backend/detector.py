from __future__ import annotations

from statistics import mean, pstdev

from PIL import Image, ImageStat


def _compression_signal(metadata: dict) -> int:
    pixels = max(1, metadata["width"] * metadata["height"])
    bytes_per_pixel = metadata["file_size_bytes"] / pixels

    if metadata["format"] == "PNG" and bytes_per_pixel < 1.2:
        return 12
    if metadata["format"] in {"JPEG", "JPG"} and bytes_per_pixel < 0.25:
        return 10
    if bytes_per_pixel > 5:
        return -4
    return 0


def _color_uniformity_signal(image: Image.Image) -> int:
    sample = image.convert("RGB").resize((96, 96))
    stat = ImageStat.Stat(sample)
    channel_stdev = mean(stat.stddev)
    channel_mean_spread = pstdev(stat.mean)

    signal = 0
    if channel_stdev < 42:
        signal += 10
    if channel_mean_spread < 18:
        signal += 6
    if channel_stdev > 75:
        signal -= 7
    return signal


def _size_signal(metadata: dict) -> int:
    width = metadata["width"]
    height = metadata["height"]

    signal = 0
    if width == height and width in {512, 768, 1024, 1536, 2048}:
        signal += 15
    if width >= 1024 and height >= 1024:
        signal += 5
    if min(width, height) < 256:
        signal -= 8
    return signal


def detect_ai_generated(image: Image.Image, metadata: dict) -> int:
    """
    Placeholder AI-generated image detector.

    Replace this with a real local PyTorch/CLIP/SigLIP model later.
    Returns an integer score from 0 to 100.
    """
    score = 35

    if not metadata["has_exif"]:
        score += 18
    else:
        score -= 6

    score += _size_signal(metadata)
    score += _compression_signal(metadata)
    score += _color_uniformity_signal(image)

    if metadata["format"] == "WEBP":
        score += 4

    return max(0, min(100, int(round(score))))
