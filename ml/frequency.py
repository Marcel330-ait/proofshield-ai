"""Frequency-domain feature extraction used by the learned fusion model."""

from __future__ import annotations

from typing import Final

import numpy as np
from PIL import Image, ImageFilter


FREQUENCY_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "log_low_energy",
    "log_mid_energy",
    "log_high_energy",
    "high_low_energy_ratio",
    "radial_decay_slope",
    "spectral_entropy",
    "spectral_flatness",
    "spectrum_mean",
    "spectrum_std",
    "spectrum_p95",
    "residual_std",
    "residual_abs_mean",
    "residual_kurtosis",
    "laplacian_energy",
    "channel_spectral_divergence",
)
FREQUENCY_FEATURE_DIM: Final[int] = len(FREQUENCY_FEATURE_NAMES)


def _safe_entropy(values: np.ndarray) -> float:
    values = np.maximum(values.astype(np.float64), 0.0)
    total = values.sum()
    if total <= 0:
        return 0.0
    probabilities = values / total
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log(probabilities)).sum())


def _radial_profile(magnitude: np.ndarray, radius: np.ndarray, bins: int = 32) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, bins + 1)
    profile = np.empty(bins, dtype=np.float64)
    for index in range(bins):
        mask = (radius >= edges[index]) & (radius < edges[index + 1])
        profile[index] = float(magnitude[mask].mean()) if np.any(mask) else 0.0
    return profile


def _laplacian_energy(gray: np.ndarray) -> float:
    center = gray[1:-1, 1:-1]
    laplacian = (
        -4.0 * center
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(np.mean(laplacian**2)) if laplacian.size else 0.0


def extract_frequency_features(image: Image.Image, size: int = 256) -> np.ndarray:
    """Return a stable, finite FFT/noise feature vector for one image.

    Features are intentionally descriptive rather than individually predictive. Their
    weights are learned only by the fusion model during training.
    """
    rgb = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    rgb_array = np.asarray(rgb, dtype=np.float32) / 255.0
    gray = 0.299 * rgb_array[..., 0] + 0.587 * rgb_array[..., 1] + 0.114 * rgb_array[..., 2]
    gray = gray - float(gray.mean())

    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.abs(spectrum).astype(np.float64)
    magnitude[size // 2, size // 2] = 0.0
    log_magnitude = np.log1p(magnitude)

    coords = np.linspace(-1.0, 1.0, size, endpoint=False)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    radius = np.sqrt(xx**2 + yy**2) / np.sqrt(2.0)
    low = magnitude[radius < 0.16]
    mid = magnitude[(radius >= 0.16) & (radius < 0.48)]
    high = magnitude[radius >= 0.48]
    low_energy = float(np.mean(low**2)) if low.size else 0.0
    mid_energy = float(np.mean(mid**2)) if mid.size else 0.0
    high_energy = float(np.mean(high**2)) if high.size else 0.0

    profile = _radial_profile(magnitude, radius)
    profile_x = np.arange(1, len(profile) + 1, dtype=np.float64)
    valid = profile > 1e-8
    slope = float(np.polyfit(np.log(profile_x[valid]), np.log(profile[valid]), 1)[0]) if valid.sum() >= 3 else 0.0

    gray_uint8 = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
    blurred = np.asarray(rgb.convert("L").filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float32) / 255.0
    residual = gray_uint8 - blurred
    residual_std = float(residual.std())
    residual_abs_mean = float(np.abs(residual).mean())
    normalized_residual = (residual - residual.mean()) / max(residual_std, 1e-6)
    residual_kurtosis = float(np.mean(normalized_residual**4))

    channel_profiles = []
    for channel in range(3):
        channel_array = rgb_array[..., channel] - float(rgb_array[..., channel].mean())
        channel_spectrum = np.abs(np.fft.fftshift(np.fft.fft2(channel_array)))
        channel_profiles.append(_radial_profile(channel_spectrum, radius))
    channel_profiles_array = np.asarray(channel_profiles)
    channel_divergence = float(channel_profiles_array.std(axis=0).mean())

    features = np.asarray(
        [
            np.log1p(low_energy),
            np.log1p(mid_energy),
            np.log1p(high_energy),
            np.log1p(high_energy) - np.log1p(low_energy),
            slope,
            _safe_entropy(magnitude),
            float(np.exp(np.log(np.maximum(magnitude, 1e-8)).mean()) / max(magnitude.mean(), 1e-8)),
            float(log_magnitude.mean()),
            float(log_magnitude.std()),
            float(np.quantile(log_magnitude, 0.95)),
            residual_std,
            residual_abs_mean,
            residual_kurtosis,
            _laplacian_energy(gray_uint8),
            channel_divergence,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

