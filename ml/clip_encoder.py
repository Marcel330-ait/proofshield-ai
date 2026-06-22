"""Frozen OpenCLIP ViT-B/32 vision branch."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from PIL import Image


class ClipVisionEncoder:
    """Local frozen CLIP image encoder.

    Downloading pretrained weights may occur once during training/bootstrap. Once
    cached, inference performs no network or external API call.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | torch.device = "cpu",
    ) -> None:
        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError("open-clip-torch is required. Install dependencies with pip install -r requirements.txt.") from exc

        self.device = torch.device(device)
        self.model_name = model_name
        self.pretrained = pretrained
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.embedding_dim = int(getattr(self.model.visual, "output_dim", 512))

    @torch.inference_mode()
    def encode_batch(self, images: Sequence[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(image.convert("RGB")) for image in images]).to(self.device)
        embeddings = self.model.encode_image(batch)
        return functional.normalize(embeddings.float(), dim=-1)

