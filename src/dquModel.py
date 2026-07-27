"""Official DQU-CIR fusion model with configurable local OpenCLIP weights."""

from __future__ import annotations

from typing import Sequence

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQU_CIR(nn.Module):
    """Adaptive fusion of DQU-CIR's unified textual and visual queries."""

    def __init__(
        self,
        hidden_dim: int = 1024,
        dropout: float = 0.5,
        *,
        clip_model: str = "ViT-H-14",
        clip_pretrained: str = "laion2B-s32B-b79K",
        clip_checkpoint: str | None = None,
        clip_cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        pretrained = clip_checkpoint or clip_pretrained
        self.clip, self.preprocess_train, self.preprocess_val = (
            open_clip.create_model_and_transforms(
                clip_model,
                pretrained=pretrained,
                cache_dir=clip_cache_dir,
            )
        )
        self.clip = self.clip.float()
        self.tokenizer = open_clip.get_tokenizer(clip_model)
        feature_dim = self._feature_dim()
        if hidden_dim != feature_dim:
            raise ValueError(
                f"hidden_dim={hidden_dim} does not match {clip_model} "
                f"output dimension {feature_dim}"
            )

        # Official implementation initializes the inverse temperature at 10.
        self.loss_weight = nn.Parameter(torch.tensor([10.0]))
        self.combiner_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.scaler_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def _feature_dim(self) -> int:
        projection = getattr(self.clip, "text_projection", None)
        if projection is not None:
            return int(projection.shape[-1])
        output_dim = getattr(self.clip.visual, "output_dim", None)
        if output_dim is None:
            raise RuntimeError("Unable to infer OpenCLIP output dimension")
        return int(output_dim)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def extract_img_fea(self, images: torch.Tensor) -> torch.Tensor:
        return self.clip.encode_image(images)

    def extract_text_fea(self, texts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(list(texts)).to(self.device)
        return self.clip.encode_text(tokens)

    def extract_query(
        self,
        textual_query: Sequence[str],
        visual_query: torch.Tensor,
        *,
        return_scaler: bool = False,
    ):
        text = F.normalize(self.extract_text_fea(textual_query), dim=-1)
        image = F.normalize(self.extract_img_fea(visual_query), dim=-1)
        combined = self.combiner_fc(torch.cat([text, image], dim=-1))
        scaler = self.scaler_fc(self.dropout(combined))
        query = F.normalize(scaler * text + (1.0 - scaler) * image, dim=-1)
        if return_scaler:
            return query, scaler
        return query

    def extract_target(self, target_img: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.extract_img_fea(target_img), dim=-1)

    def compute_loss(
        self,
        textual_query: Sequence[str],
        visual_query: torch.Tensor,
        target_img: torch.Tensor,
    ):
        query = self.extract_query(textual_query, visual_query)
        target = self.extract_target(target_img)
        return {"ranking": self.ranking_nce_loss(query, target)}

    def ranking_nce_loss(
        self, query: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        logits = self.loss_weight * (query @ target.t())
        labels = torch.arange(logits.shape[0], device=logits.device)
        # Keep the paper/repository's one-way batch classification objective.
        return F.cross_entropy(logits, labels)
