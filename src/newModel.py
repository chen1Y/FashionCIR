"""Retrieval model with raw-text, structured-description, and image gating."""

from __future__ import annotations

import math
from typing import Sequence

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureGate(nn.Module):
    """Predict a per-sample interpolation weight with a controlled initial value."""

    def __init__(self, feature_dim: int, dropout: float, initial_weight: float) -> None:
        super().__init__()
        if not 0.0 < initial_weight < 1.0:
            raise ValueError("initial_weight must be strictly between 0 and 1")
        hidden_dim = max(feature_dim // 2, 128)
        self.network = nn.Sequential(
            nn.Linear(feature_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        final = self.network[-1]
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, math.log(initial_weight / (1.0 - initial_weight)))

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [first, second, torch.abs(first - second), first * second], dim=-1
        )
        return torch.sigmoid(self.network(features))


class DQU_CIR(nn.Module):
    """Compose a FashionIQ query without loading Qwen during retrieval training.

    Qwen is used offline by ``generate_structured_edits.py``.  At training and
    inference time this model consumes the original modification text and the
    pre-generated target description, encodes both with the same frozen CLIP
    text tower, and learns two gates:

    1. target-description weight inside the text representation;
    2. reference-image weight inside the final query representation.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        num_heads: int = 8,
        *,
        clip_model: str = "ViT-H-14",
        clip_pretrained: str = "laion2B-s32B-b79K",
        clip_checkpoint: str | None = None,
        clip_cache_dir: str | None = None,
        freeze_clip: bool = True,
        initial_target_weight: float = 0.25,
        initial_image_weight: float = 0.25,
        temperature: float = 10.0,
    ) -> None:
        super().__init__()
        del hidden_dim, num_heads  # Kept in the signature for old command compatibility.

        pretrained = clip_checkpoint or clip_pretrained
        self.clip, self.preprocess_train, self.preprocess_val = (
            open_clip.create_model_and_transforms(
            clip_model,
            pretrained=pretrained,
            cache_dir=clip_cache_dir,
            )
        )
        self.tokenizer = open_clip.get_tokenizer(clip_model)
        self.clip_model_name = clip_model
        self.freeze_clip = freeze_clip
        if freeze_clip:
            self.clip.requires_grad_(False)

        feature_dim = self._infer_feature_dim()
        self.feature_dim = feature_dim
        self.text_gate = FeatureGate(feature_dim, dropout, initial_target_weight)
        self.image_gate = FeatureGate(feature_dim, dropout, initial_image_weight)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(temperature)))

    def _infer_feature_dim(self) -> int:
        projection = getattr(self.clip, "text_projection", None)
        if projection is not None:
            return int(projection.shape[-1])
        visual = getattr(self.clip, "visual", None)
        output_dim = getattr(visual, "output_dim", None)
        if output_dim is None:
            raise RuntimeError("Unable to infer OpenCLIP output dimension")
        return int(output_dim)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_clip:
            self.clip.eval()
        return self

    def extract_img_fea(self, images: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad() if self.freeze_clip else torch.enable_grad()
        with context:
            features = self.clip.encode_image(images)
        return F.normalize(features.float(), dim=-1)

    def extract_text_fea(self, texts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(list(texts)).to(self.device)
        context = torch.no_grad() if self.freeze_clip else torch.enable_grad()
        with context:
            features = self.clip.encode_text(tokens)
        return F.normalize(features.float(), dim=-1)

    @staticmethod
    def _as_mask(
        target_description_mask: torch.Tensor | Sequence[bool] | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if target_description_mask is None:
            return torch.ones(batch_size, 1, device=device, dtype=torch.bool)
        mask = torch.as_tensor(target_description_mask, device=device, dtype=torch.bool)
        return mask.reshape(batch_size, 1)

    def extract_query(
        self,
        textual_query: Sequence[str],
        target_description: Sequence[str],
        visual_query: torch.Tensor,
        target_description_mask: torch.Tensor | Sequence[bool] | None = None,
        *,
        return_diagnostics: bool = False,
    ):
        raw_text = self.extract_text_fea(textual_query)
        target_text = self.extract_text_fea(target_description)
        image = self.extract_img_fea(visual_query)
        mask = self._as_mask(target_description_mask, len(textual_query), image.device)

        predicted_target_weight = self.text_gate(raw_text, target_text)
        target_weight = torch.where(
            mask,
            predicted_target_weight,
            torch.zeros_like(predicted_target_weight),
        )
        text = F.normalize(
            (1.0 - target_weight) * raw_text + target_weight * target_text,
            dim=-1,
        )

        image_weight = self.image_gate(text, image)
        query = F.normalize(
            (1.0 - image_weight) * text + image_weight * image,
            dim=-1,
        )
        if not return_diagnostics:
            return query
        diagnostics = {
            "target_weight": target_weight.detach(),
            "image_weight": image_weight.detach(),
            "structured_coverage": mask.float().mean().detach(),
        }
        return query, diagnostics

    def extract_target(self, target_img: torch.Tensor) -> torch.Tensor:
        return self.extract_img_fea(target_img)

    def compute_loss(
        self,
        textual_query: Sequence[str],
        target_description: Sequence[str],
        visual_query: torch.Tensor,
        target_img: torch.Tensor,
        target_description_mask: torch.Tensor | Sequence[bool] | None = None,
    ) -> dict[str, torch.Tensor]:
        query, diagnostics = self.extract_query(
            textual_query,
            target_description,
            visual_query,
            target_description_mask,
            return_diagnostics=True,
        )
        target = self.extract_target(target_img)
        ranking = self.ranking_nce_loss(query, target)
        return {
            "ranking": ranking,
            "target_weight": diagnostics["target_weight"].mean(),
            "image_weight": diagnostics["image_weight"].mean(),
            "structured_coverage": diagnostics["structured_coverage"],
        }

    def ranking_nce_loss(self, query: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * query @ target.t()
        labels = torch.arange(logits.shape[0], device=logits.device)
        # Symmetric retrieval loss gives every image and every query a learning signal.
        return 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.t(), labels)
        )
