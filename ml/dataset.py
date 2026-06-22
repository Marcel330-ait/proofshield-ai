"""Manifest-driven dataset, labels, and corruption-aware training augmentation."""

from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset

from .metadata import summarize_metadata


@dataclass(frozen=True)
class ManifestRecord:
    path: Path
    label: int
    split: str
    source: str


def load_manifest(manifest_path: str | Path, split: str | None = None) -> list[ManifestRecord]:
    manifest_path = Path(manifest_path)
    records: list[ManifestRecord] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "label", "split", "source"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Manifest must contain path,label,split,source columns.")
        for line_number, row in enumerate(reader, start=2):
            row_split = str(row["split"]).strip().lower()
            if split and row_split != split.lower():
                continue
            try:
                label = int(row["label"])
            except ValueError as exc:
                raise ValueError(f"Invalid label at manifest line {line_number}.") from exc
            if label not in (0, 1):
                raise ValueError(f"Label at manifest line {line_number} must be 0 (real) or 1 (AI).")
            path = Path(row["path"])
            if not path.is_absolute():
                path = (manifest_path.parent / path).resolve()
            records.append(ManifestRecord(path=path, label=label, split=row_split, source=str(row["source"]).strip()))
    if not records:
        raise ValueError(f"No records found for split '{split}'.")
    return records


class ForensicAugmentation:
    """PIL-only corruptions approximating common distribution shift."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    @staticmethod
    def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        with Image.open(io.BytesIO(buffer.getvalue())) as decoded:
            return decoded.convert("RGB").copy()

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if not self.enabled:
            return image

        width, height = image.size
        if random.random() < 0.55:
            scale = random.uniform(0.45, 0.9)
            reduced = image.resize((max(32, int(width * scale)), max(32, int(height * scale))), Image.Resampling.LANCZOS)
            image = reduced.resize((width, height), Image.Resampling.LANCZOS)
        if random.random() < 0.35:
            crop_scale = random.uniform(0.72, 0.95)
            crop_w, crop_h = int(width * crop_scale), int(height * crop_scale)
            left = random.randint(0, max(0, width - crop_w))
            top = random.randint(0, max(0, height - crop_h))
            image = image.crop((left, top, left + crop_w, top + crop_h)).resize((width, height), Image.Resampling.LANCZOS)
        if random.random() < 0.65:
            image = self._jpeg_roundtrip(image, quality=random.randint(35, 92))
        if random.random() < 0.22:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.25, 1.4)))
        if random.random() < 0.25:
            pixels = np.asarray(image, dtype=np.float32)
            pixels += np.random.normal(0.0, random.uniform(1.5, 8.0), pixels.shape)
            image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB")
        if random.random() < 0.18:
            image = ImageOps.expand(image, border=random.randint(1, 5), fill=(245, 245, 245)).resize((width, height), Image.Resampling.LANCZOS)
        return image


class ForensicImageDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[ManifestRecord], augment: bool = False) -> None:
        self.records = records
        self.augment = ForensicAugmentation(enabled=augment)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        file_bytes = record.path.read_bytes()
        with Image.open(io.BytesIO(file_bytes)) as opened:
            opened.load()
            original = opened.copy()
        metadata = summarize_metadata(original, file_bytes)
        return {
            "image": self.augment(original),
            "label": record.label,
            "metadata": metadata,
            "source": record.source,
            "path": str(record.path),
        }


def collate_forensics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [sample["image"] for sample in samples],
        "labels": [sample["label"] for sample in samples],
        "metadata": [sample["metadata"] for sample in samples],
        "sources": [sample["source"] for sample in samples],
        "paths": [sample["path"] for sample in samples],
    }

