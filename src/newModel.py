"""DQU-CIR with a confidence-aware Qwen structured-text residual."""

from __future__ import annotations

from typing import Sequence

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuredGate(nn.Module):
    """Predict sample-specific residual weights from two CLIP text views."""

    def __init__(self, feature_dim: int, dropout: float) -> None:
        super().__init__()
        del dropout  # Avoid advancing DQU-CIR's dropout RNG at zero residual.
        hidden_dim = max(feature_dim // 2, 128)
        self.network = nn.Sequential(
            nn.Linear(feature_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Begin with a neutral 0.5 gate. The global residual strength below is
        # exactly zero, so the complete model initially reproduces DQU-CIR.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, baseline: torch.Tensor, structured: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [
                baseline,
                structured,
                torch.abs(baseline - structured),
                baseline * structured,
            ],
            dim=-1,
        )
        return torch.sigmoid(self.network(features))


class DQU_CIR(nn.Module):
    """Official DQU-CIR query fusion plus an additive structured text branch.

    Qwen runs offline. During retrieval this module encodes both the official
    DQU textual query and a controlled natural-language rendering of Qwen JSON.
    A zero-initialized residual strength makes the initial query exactly equal
    to DQU-CIR, while validation masks and Qwen confidence bound the new branch.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        dropout: float = 0.5,
        *,
        clip_model: str = "ViT-H-14",
        clip_pretrained: str = "laion2B-s32B-b79K",
        clip_checkpoint: str | None = None,
        clip_cache_dir: str | None = None,
        freeze_clip: bool = False,
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
        self.freeze_clip = freeze_clip
        if freeze_clip:
            self.clip.requires_grad_(False)

        feature_dim = self._feature_dim()
        if hidden_dim != feature_dim:
            raise ValueError(
                f"hidden_dim={hidden_dim} does not match {clip_model} "
                f"output dimension {feature_dim}"
            )

        # Names and initialization match dquModel.py so an official DQU
        # checkpoint can be loaded directly into the shared baseline modules.
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

        self.structured_gate = StructuredGate(feature_dim, dropout)
        self.structured_strength = nn.Parameter(torch.tensor(0.0))

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

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_clip:
            self.clip.eval()
        return self

    def extract_img_fea(self, images: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad() if self.freeze_clip else torch.enable_grad()
        with context:
            return self.clip.encode_image(images)

    def extract_text_fea(self, texts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(list(texts)).to(self.device)
        context = torch.no_grad() if self.freeze_clip else torch.enable_grad()
        with context:
            return self.clip.encode_text(tokens)

    @staticmethod
    def _column(
        values,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.as_tensor(values, device=device, dtype=dtype).reshape(batch_size, 1)

    def extract_query(
        self,
        textual_query: Sequence[str],
        structured_text: Sequence[str],
        visual_query: torch.Tensor,
        structured_mask=None,
        structured_confidence=None,
        *,
        return_diagnostics: bool = False,
    ):
        baseline = F.normalize(self.extract_text_fea(textual_query), dim=-1)
        structured = F.normalize(self.extract_text_fea(structured_text), dim=-1)
        image = F.normalize(self.extract_img_fea(visual_query), dim=-1)
        batch_size = len(textual_query)

        if structured_mask is None:
            structured_mask = torch.ones(batch_size, device=image.device)
        if structured_confidence is None:
            structured_confidence = torch.ones(batch_size, device=image.device)
        mask = self._column(
            structured_mask, batch_size, image.device, baseline.dtype
        )
        confidence = self._column(
            structured_confidence, batch_size, image.device, baseline.dtype
        ).clamp(0.0, 1.0)
        predicted_gate = self.structured_gate(baseline, structured)
        global_strength = torch.tanh(self.structured_strength)
        structured_weight = global_strength * predicted_gate * confidence * mask
        text = F.normalize(
            baseline + structured_weight * (structured - baseline),
            dim=-1,
        )

        combined = self.combiner_fc(torch.cat([text, image], dim=-1))
        dqu_text_weight = self.scaler_fc(self.dropout(combined))
        query = F.normalize(
            dqu_text_weight * text + (1.0 - dqu_text_weight) * image,
            dim=-1,
        )
        if not return_diagnostics:
            return query
        diagnostics = {
            "structured_weight": structured_weight.detach(),
            "predicted_structured_gate": predicted_gate.detach(),
            "structured_strength": global_strength.detach(),
            "dqu_text_weight": dqu_text_weight.detach(),
            "structured_coverage": mask.mean().detach(),
            "mean_confidence": (confidence * mask).sum().detach()
            / mask.sum().clamp_min(1.0),
        }
        return query, diagnostics

    def extract_target(self, target_img: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.extract_img_fea(target_img), dim=-1)

    def compute_loss(
        self,
        textual_query: Sequence[str],
        structured_text: Sequence[str],
        visual_query: torch.Tensor,
        target_img: torch.Tensor,
        structured_mask=None,
        structured_confidence=None,
    ) -> dict[str, torch.Tensor]:
        query, diagnostics = self.extract_query(
            textual_query,
            structured_text,
            visual_query,
            structured_mask,
            structured_confidence,
            return_diagnostics=True,
        )
        target = self.extract_target(target_img)
        return {
            "ranking": self.ranking_nce_loss(query, target),
            **{name: value.mean() for name, value in diagnostics.items()},
        }

    def ranking_nce_loss(
        self, query: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        logits = self.loss_weight * (query @ target.t())
        labels = torch.arange(logits.shape[0], device=logits.device)
        # Preserve DQU-CIR's one-way batch classification objective.
        return F.cross_entropy(logits, labels)
