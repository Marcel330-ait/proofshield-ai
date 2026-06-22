"""Reproducible training entry point for the frozen-CLIP fusion detector."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .clip_encoder import ClipVisionEncoder
from .dataset import ForensicImageDataset, collate_forensics, load_manifest
from .evaluate import binary_metrics
from .frequency import extract_frequency_features
from .metadata import extract_metadata_features
from .model import ModelConfig, build_fusion, save_checkpoint


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base_loss = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probabilities, 1.0 - probabilities)
        return ((1.0 - pt).pow(self.gamma) * base_loss).mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_features(encoder: ClipVisionEncoder, images, metadata, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vision = encoder.encode_batch(images)
    frequency = torch.tensor(np.stack([extract_frequency_features(image) for image in images]), dtype=torch.float32, device=device)
    meta = torch.tensor(np.asarray([extract_metadata_features(item) for item in metadata]), dtype=torch.float32, device=device)
    return vision, frequency, meta


@torch.inference_mode()
def fit_standardizer(model, encoder, loader: DataLoader, device: torch.device) -> None:
    collected: list[torch.Tensor] = []
    for batch in loader:
        vision, frequency, metadata = make_features(encoder, batch["images"], batch["metadata"], device)
        collected.append(model.join_features(vision, frequency, metadata).cpu())
    model.standardizer.fit(torch.cat(collected, dim=0).to(device))


def fit_temperature(model, logits: torch.Tensor, labels: torch.Tensor, device: torch.device) -> None:
    model.calibrator.to(device)
    optimizer = torch.optim.LBFGS(model.calibrator.parameters(), lr=0.1, max_iter=80, line_search_fn="strong_wolfe")
    criterion = nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = criterion(model.calibrator(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)


@torch.no_grad()
def validate(model, encoder, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    model.eval()
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    for batch in loader:
        vision, frequency, metadata = make_features(encoder, batch["images"], batch["metadata"], device)
        logits_all.append(model(vision, frequency, metadata).cpu())
        labels_all.append(torch.tensor(batch["labels"], dtype=torch.float32))
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    probabilities = torch.sigmoid(model.calibrator(logits.to(device))).cpu().numpy()
    return binary_metrics(probabilities, labels.numpy()), logits, labels


def train(args) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device)
    train_records = load_manifest(args.manifest, split="train")
    val_records = load_manifest(args.manifest, split="val")
    train_loader = DataLoader(ForensicImageDataset(train_records, augment=True), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_forensics, pin_memory=device.type == "cuda")
    stats_loader = DataLoader(ForensicImageDataset(train_records, augment=False), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_forensics)
    val_loader = DataLoader(ForensicImageDataset(val_records, augment=False), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_forensics)

    encoder = ClipVisionEncoder(args.clip_model, args.clip_pretrained, device)
    config = ModelConfig(clip_model=args.clip_model, clip_pretrained=args.clip_pretrained, vision_dim=encoder.embedding_dim)
    model = build_fusion(config).to(device)
    fit_standardizer(model, encoder, stats_loader, device)
    criterion = FocalLoss() if args.focal_loss else nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best: dict = {"roc_auc": -1.0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        sample_count = 0
        for batch in train_loader:
            vision, frequency, metadata = make_features(encoder, batch["images"], batch["metadata"], device)
            labels = torch.tensor(batch["labels"], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(vision, frequency, metadata)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item()) * len(labels)
            sample_count += len(labels)

        validation_metrics, validation_logits, validation_labels = validate(model, encoder, val_loader, device)
        fit_temperature(model, validation_logits.to(device), validation_labels.to(device), device)
        validation_metrics, _, _ = validate(model, encoder, val_loader, device)
        summary = {"epoch": epoch, "train_loss": epoch_loss / max(sample_count, 1), **validation_metrics, "temperature": model.calibrator.value()}
        print(json.dumps(summary, allow_nan=False))
        if validation_metrics["roc_auc"] > best["roc_auc"]:
            best = summary
            save_checkpoint(
                args.output,
                model,
                config,
                metrics=best,
                training_data={"manifest": str(args.manifest), "train_samples": len(train_records), "val_samples": len(val_records), "seed": args.seed},
            )
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ProofShield AI v2 fusion model.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/model.pt"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-loss", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    best = train(args)
    print(f"Saved best calibrated checkpoint to {args.output}: {json.dumps(best, allow_nan=False)}")


if __name__ == "__main__":
    main()

