"""Learned feature fusion and probability calibration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class FeatureStandardizer(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(feature_dim))
        self.register_buffer("std", torch.ones(feature_dim))

    @torch.no_grad()
    def fit(self, features: torch.Tensor) -> None:
        self.mean.copy_(features.mean(dim=0))
        self.std.copy_(features.std(dim=0, unbiased=False).clamp_min(1e-6))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean) / self.std.clamp_min(1e-6)


class FusionMLP(nn.Module):
    """The requested 512 -> 128 MLP, returning logits for stable BCE training."""

    def __init__(self, input_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class TemperatureScaler(nn.Module):
    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(initial_temperature))))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(0.05, 10.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    @torch.no_grad()
    def value(self) -> float:
        return float(self.temperature.item())


@dataclass(frozen=True)
class FeatureSlices:
    vision: slice
    frequency: slice
    metadata: slice


class MultimodalFusion(nn.Module):
    def __init__(self, vision_dim: int, frequency_dim: int, metadata_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.vision_dim = vision_dim
        self.frequency_dim = frequency_dim
        self.metadata_dim = metadata_dim
        self.input_dim = vision_dim + frequency_dim + metadata_dim
        self.standardizer = FeatureStandardizer(self.input_dim)
        self.classifier = FusionMLP(self.input_dim, dropout=dropout)
        self.calibrator = TemperatureScaler()

    @property
    def slices(self) -> FeatureSlices:
        return FeatureSlices(
            vision=slice(0, self.vision_dim),
            frequency=slice(self.vision_dim, self.vision_dim + self.frequency_dim),
            metadata=slice(self.vision_dim + self.frequency_dim, self.input_dim),
        )

    def join_features(self, vision: torch.Tensor, frequency: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        return torch.cat((vision, frequency, metadata), dim=-1)

    def forward(self, vision: torch.Tensor, frequency: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        features = self.join_features(vision, frequency, metadata)
        return self.classifier(self.standardizer(features))

    def probability(self, vision: torch.Tensor, frequency: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.calibrator(self.forward(vision, frequency, metadata)))

    def _mean_branch(self, branch: str, batch_size: int, device: torch.device) -> torch.Tensor:
        branch_slice = getattr(self.slices, branch)
        return self.standardizer.mean[branch_slice].to(device).unsqueeze(0).expand(batch_size, -1)

    @torch.no_grad()
    def branch_probabilities(self, vision: torch.Tensor, frequency: torch.Tensor, metadata: torch.Tensor) -> dict[str, torch.Tensor]:
        """Conditional branch-only probabilities using learned training-mean baselines."""
        batch_size = vision.shape[0]
        baseline_vision = self._mean_branch("vision", batch_size, vision.device)
        baseline_frequency = self._mean_branch("frequency", batch_size, vision.device)
        baseline_metadata = self._mean_branch("metadata", batch_size, vision.device)
        return {
            "vision": self.probability(vision, baseline_frequency, baseline_metadata),
            "frequency": self.probability(baseline_vision, frequency, baseline_metadata),
            "metadata": self.probability(baseline_vision, baseline_frequency, metadata),
        }

    @torch.no_grad()
    def branch_ablation(self, vision: torch.Tensor, frequency: torch.Tensor, metadata: torch.Tensor) -> dict[str, torch.Tensor]:
        """Learned contribution measured by ablation against the training mean."""
        full = self.probability(vision, frequency, metadata)
        batch_size = vision.shape[0]
        baselines = {
            "vision": self._mean_branch("vision", batch_size, vision.device),
            "frequency": self._mean_branch("frequency", batch_size, vision.device),
            "metadata": self._mean_branch("metadata", batch_size, vision.device),
        }
        scores: dict[str, torch.Tensor] = {}
        for branch, replacement in baselines.items():
            values = {"vision": vision, "frequency": frequency, "metadata": metadata}
            values[branch] = replacement
            scores[branch] = full - self.probability(values["vision"], values["frequency"], values["metadata"])
        return scores
