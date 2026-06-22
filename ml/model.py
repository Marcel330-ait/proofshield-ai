"""Checkpoint contract for the ProofShield AI v2 multimodal model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .frequency import FREQUENCY_FEATURE_DIM
from .fusion import MultimodalFusion
from .metadata import METADATA_FEATURE_DIM


@dataclass(frozen=True)
class ModelConfig:
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    vision_dim: int = 512
    frequency_dim: int = FREQUENCY_FEATURE_DIM
    metadata_dim: int = METADATA_FEATURE_DIM
    dropout: float = 0.3
    version: str = "2.0"


def build_fusion(config: ModelConfig) -> MultimodalFusion:
    return MultimodalFusion(
        vision_dim=config.vision_dim,
        frequency_dim=config.frequency_dim,
        metadata_dim=config.metadata_dim,
        dropout=config.dropout,
    )


def save_checkpoint(
    path: str | Path,
    model: MultimodalFusion,
    config: ModelConfig,
    metrics: dict[str, float],
    training_data: dict[str, Any],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "model_config": asdict(config),
            "state_dict": model.state_dict(),
            "metrics": metrics,
            "training_data": training_data,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str | torch.device) -> tuple[MultimodalFusion, ModelConfig, dict[str, Any]]:
    artifact = torch.load(path, map_location=device, weights_only=False)
    if artifact.get("format_version") != 2:
        raise ValueError("Unsupported checkpoint format. Train or export a ProofShield AI v2 checkpoint.")
    config = ModelConfig(**artifact["model_config"])
    model = build_fusion(config).to(device)
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.eval()
    return model, config, artifact

