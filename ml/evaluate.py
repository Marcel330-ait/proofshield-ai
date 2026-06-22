"""Evaluation, calibration error, and mandatory corruption robustness report."""

from __future__ import annotations

import argparse
import io
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from .clip_encoder import ClipVisionEncoder
from .dataset import ForensicImageDataset, collate_forensics, load_manifest
from .frequency import extract_frequency_features
from .metadata import extract_metadata_features
from .model import load_checkpoint


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (probabilities >= lower) & (probabilities < upper if index < bins - 1 else probabilities <= upper)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(probabilities[mask].mean()) - float(labels[mask].mean()))
    return ece


def binary_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "ece": expected_calibration_error(probabilities, labels),
    }
    metrics["roc_auc"] = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float("nan")
    metrics["average_precision"] = float(average_precision_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float("nan")
    return metrics


def _jpeg(image: Image.Image) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=35)
    with Image.open(io.BytesIO(buffer.getvalue())) as decoded:
        return decoded.convert("RGB").copy()


def _resize(image: Image.Image) -> Image.Image:
    width, height = image.size
    return image.resize((max(32, width // 2), max(32, height // 2)), Image.Resampling.LANCZOS).resize((width, height), Image.Resampling.LANCZOS)


def _crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    left, top = int(width * 0.12), int(height * 0.12)
    return image.crop((left, top, width - left, height - top)).resize((width, height), Image.Resampling.LANCZOS)


def _screenshot(image: Image.Image) -> Image.Image:
    width, height = image.size
    bordered = Image.new("RGB", (width + 24, height + 48), (238, 238, 238))
    bordered.paste(image.convert("RGB"), (12, 36))
    return _jpeg(bordered).resize((width, height), Image.Resampling.LANCZOS)


CORRUPTIONS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "jpeg_q35": _jpeg,
    "resize_50pct": _resize,
    "center_crop": _crop,
    "gaussian_blur": lambda image: image.filter(ImageFilter.GaussianBlur(radius=1.25)),
    "screenshot": _screenshot,
}


@torch.inference_mode()
def predict_loader(model, encoder, loader: DataLoader, device: torch.device, transform: Callable[[Image.Image], Image.Image] | None = None) -> tuple[np.ndarray, np.ndarray]:
    all_probabilities: list[float] = []
    all_labels: list[int] = []
    for batch in loader:
        images = [transform(image) if transform else image for image in batch["images"]]
        vision = encoder.encode_batch(images)
        frequency = torch.tensor(np.stack([extract_frequency_features(image) for image in images]), dtype=torch.float32, device=device)
        metadata = torch.tensor(np.asarray([extract_metadata_features(item) for item in batch["metadata"]]), dtype=torch.float32, device=device)
        probabilities = model.probability(vision, frequency, metadata)
        all_probabilities.extend(probabilities.detach().cpu().tolist())
        all_labels.extend(batch["labels"])
    return np.asarray(all_probabilities), np.asarray(all_labels)


def run_evaluation(checkpoint: Path, manifest: Path, output: Path, batch_size: int, device_name: str) -> dict:
    device = torch.device(device_name)
    model, config, _ = load_checkpoint(checkpoint, device)
    encoder = ClipVisionEncoder(config.clip_model, config.clip_pretrained, device)
    records = load_manifest(manifest, split="test")
    loader = DataLoader(ForensicImageDataset(records, augment=False), batch_size=batch_size, shuffle=False, collate_fn=collate_forensics)
    base_probabilities, labels = predict_loader(model, encoder, loader, device)
    clean = binary_metrics(base_probabilities, labels)
    robustness = {}
    for name, transform in CORRUPTIONS.items():
        probabilities, corrupted_labels = predict_loader(model, encoder, loader, device, transform)
        current = binary_metrics(probabilities, corrupted_labels)
        robustness[name] = {**current, "accuracy_drop": clean["accuracy"] - current["accuracy"], "roc_auc_drop": clean["roc_auc"] - current["roc_auc"]}
    report = {"checkpoint": str(checkpoint), "config": config.__dict__, "clean": clean, "robustness": robustness}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    report = run_evaluation(args.checkpoint, args.manifest, args.output, args.batch_size, args.device)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

