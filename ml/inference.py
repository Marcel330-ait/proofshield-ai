"""Offline inference with learned branch contribution explanations."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .clip_encoder import ClipVisionEncoder
from .frequency import extract_frequency_features
from .metadata import extract_metadata_features, summarize_metadata
from .model import load_checkpoint


class ModelNotReadyError(RuntimeError):
    pass


def risk_level(probability: float) -> str:
    if probability < 0.40:
        return "Low"
    if probability < 0.70:
        return "Medium"
    return "High"


class LocalInferenceEngine:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise ModelNotReadyError(f"Validated model checkpoint not found: {self.checkpoint_path}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.config, self.artifact = load_checkpoint(self.checkpoint_path, self.device)
        self.encoder = ClipVisionEncoder(self.config.clip_model, self.config.clip_pretrained, self.device)

    @torch.inference_mode()
    def analyze_bytes(self, file_bytes: bytes) -> dict:
        with Image.open(io.BytesIO(file_bytes)) as opened:
            opened.load()
            image = opened.copy()
        metadata = summarize_metadata(image, file_bytes)
        vision = self.encoder.encode_batch([image])
        frequency = torch.tensor(extract_frequency_features(image)[None, :], dtype=torch.float32, device=self.device)
        meta = torch.tensor(np.asarray([extract_metadata_features(metadata)]), dtype=torch.float32, device=self.device)
        probability_tensor = self.model.probability(vision, frequency, meta)
        probability = float(probability_tensor.item())
        branch_probabilities = self.model.branch_probabilities(vision, frequency, meta)
        ablation = self.model.branch_ablation(vision, frequency, meta)
        contributions = {name: float(value.item()) for name, value in ablation.items()}
        ranking = sorted(contributions, key=lambda name: abs(contributions[name]), reverse=True)
        primary = ranking[0]
        explanations = [
            f"The calibrated multimodal model estimates {probability * 100:.1f}% AI-generated risk; this is probabilistic, not a authenticity verdict.",
            f"The frozen CLIP, frequency, and metadata branches were fused by a trained classifier. The largest learned contribution for this image was the {primary} branch.",
        ]
        if abs(contributions["metadata"]) > 0.02:
            explanations.append("Metadata was treated as an auxiliary learned feature and did not independently determine the result.")
        confidence = abs(probability - 0.5) * 2.0
        return {
            "ai_probability": round(probability * 100.0, 2),
            "risk_level": risk_level(probability),
            "confidence": round(confidence, 4),
            "signals": {
                "vision_score": round(float(branch_probabilities["vision"].item()), 4),
                "frequency_score": round(float(branch_probabilities["frequency"].item()), 4),
                "metadata_score": round(float(branch_probabilities["metadata"].item()), 4),
                "learned_contributions": {name: round(value, 4) for name, value in contributions.items()},
            },
            "explanation": explanations,
            "recommendations": [
                "Use the original file and trusted provenance records for any high-impact decision.",
                "Review the model version and held-out evaluation metrics before relying on this estimate.",
            ],
            "metadata": metadata,
            "model": {"version": self.config.version, "checkpoint_metrics": self.artifact.get("metrics", {})},
            "disclaimer": "This is an AI-generated risk estimate from a trained model. It does not establish whether an image is real, fake, lawful, or unlawful.",
        }
