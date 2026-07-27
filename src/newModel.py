"""A protected DQU-CIR backbone with a Qwen structured-text adapter."""

from __future__ import annotations

from typing import Sequence

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuredAdapter(nn.Module):
    """Learn a signed residual instead of directly trusting CLIP text geometry."""

    def __init__(self, feature_dim: int, rank: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, rank),
            nn.GELU(),
            nn.Linear(rank, feature_dim),
        )
        # Exact DQU-CIR at initialization, while the final layer still receives
        # gradients on the first step.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self, baseline: torch.Tensor, structured: torch.Tensor
    ) -> torch.Tensor:
        return self.network(structured - baseline)


class StructuredGate(nn.Module):
    """Predict whether structured text is useful for this image-query pair."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.network[-2].weight)
        nn.init.zeros_(self.network[-2].bias)

    def forward(
        self,
        baseline: torch.Tensor,
        structured: torch.Tensor,
        image: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        text_similarity = (baseline * structured).sum(dim=-1, keepdim=True)
        baseline_image = (baseline * image).sum(dim=-1, keepdim=True)
        structured_image = (structured * image).sum(dim=-1, keepdim=True)
        distance = (structured - baseline).pow(2).mean(dim=-1, keepdim=True).sqrt()
        features = torch.cat(
            [
                text_similarity,
                baseline_image,
                structured_image,
                distance,
                confidence,
            ],
            dim=-1,
        )
        return self.network(features)


class DQU_CIR(nn.Module):
    """DQU-CIR plus a bounded, confidence-aware structured feature adapter."""

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
        adapter_rank: int = 256,
        max_structured_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if adapter_rank <= 0:
            raise ValueError("adapter_rank must be positive")
        if not 0.0 < max_structured_weight <= 1.0:
            raise ValueError("max_structured_weight must be in (0, 1]")
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

        # These names and initializations match dquModel.py.
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

        self.structured_adapter = StructuredAdapter(feature_dim, adapter_rank)
        self.structured_gate = StructuredGate()
        self.max_structured_weight = float(max_structured_weight)

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

    def freeze_dqu_backbone(self) -> None:
        """Freeze the verified retrieval model and train only Qwen modules."""
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("structured_"))
        self.freeze_clip = True
        self.clip.eval()

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
    def _column(values, batch_size, device, dtype) -> torch.Tensor:
        return torch.as_tensor(values, device=device, dtype=dtype).reshape(
            batch_size, 1
        )

    def _dqu_fusion(self, text, image):
        combined = self.combiner_fc(torch.cat([text, image], dim=-1))
        text_weight = self.scaler_fc(self.dropout(combined))
        query = F.normalize(
            text_weight * text + (1.0 - text_weight) * image,
            dim=-1,
        )
        return query, text_weight

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

        predicted_gate = self.structured_gate(
            baseline, structured, image, confidence
        )
        structured_weight = (
            self.max_structured_weight * predicted_gate * confidence * mask
        )
        residual = self.structured_adapter(baseline, structured)
        adapted_text = F.normalize(
            baseline + structured_weight * residual,
            dim=-1,
        )
        query, dqu_text_weight = self._dqu_fusion(adapted_text, image)

        if not return_diagnostics:
            return query
        diagnostics = {
            "structured_weight": structured_weight.detach(),
            "predicted_structured_gate": predicted_gate.detach(),
            "adapter_residual_norm": residual.norm(dim=-1).detach(),
            "dqu_text_weight": dqu_text_weight.detach(),
            "structured_coverage": mask.mean().detach(),
            "mean_confidence": (confidence * mask).sum().detach()
            / mask.sum().clamp_min(1.0),
            "text_drift": (
                1.0 - (adapted_text * baseline).sum(dim=-1)
            ).clamp_min(0.0).detach(),
            "baseline_text": baseline,
            "structured_text": structured,
            "image_feature": image,
            "adapted_text": adapted_text,
            "predicted_gate_for_loss": predicted_gate,
            "mask_for_loss": mask,
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
        preservation_weight: float = 1.0,
        gate_supervision_weight: float = 0.2,
        gate_teacher_temperature: float = 0.1,
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
        ranking = self.ranking_nce_loss(query, target)
        baseline_text = diagnostics.pop("baseline_text").detach()
        structured_text = diagnostics.pop("structured_text").detach()
        image_feature = diagnostics.pop("image_feature").detach()
        adapted_text = diagnostics.pop("adapted_text")
        predicted_gate = diagnostics.pop("predicted_gate_for_loss")
        valid_mask = diagnostics.pop("mask_for_loss")
        preservation = (
            1.0 - (adapted_text * baseline_text).sum(dim=-1)
        ).clamp_min(0.0).mean()

        # A detached per-sample teacher asks a concrete retrieval question:
        # would replacing the raw modification with Qwen structured text lower
        # this sample's in-batch NCE loss?  This prevents the gate from simply
        # increasing on every sample as the adapter overfits.
        with torch.no_grad():
            text_weight = diagnostics["dqu_text_weight"].reshape(-1, 1)
            baseline_query = F.normalize(
                text_weight * baseline_text
                + (1.0 - text_weight) * image_feature,
                dim=-1,
            )
            structured_query = F.normalize(
                text_weight * structured_text
                + (1.0 - text_weight) * image_feature,
                dim=-1,
            )
            labels = torch.arange(target.shape[0], device=target.device)
            baseline_loss = F.cross_entropy(
                self.loss_weight.detach() * (baseline_query @ target.t()),
                labels,
                reduction="none",
            )
            structured_loss = F.cross_entropy(
                self.loss_weight.detach() * (structured_query @ target.t()),
                labels,
                reduction="none",
            )
            teacher_gate = torch.sigmoid(
                (baseline_loss - structured_loss)
                / max(float(gate_teacher_temperature), 1e-6)
            ).reshape(-1, 1)

        gate_bce = F.binary_cross_entropy(
            predicted_gate.clamp(1e-6, 1.0 - 1e-6),
            teacher_gate,
            reduction="none",
        )
        gate_supervision = (gate_bce * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        total = (
            ranking
            + preservation_weight * preservation
            + gate_supervision_weight * gate_supervision
        )
        return {
            "loss": total,
            "ranking": ranking,
            "preservation": preservation,
            "gate_supervision": gate_supervision,
            "teacher_gate": (teacher_gate * valid_mask).sum()
            / valid_mask.sum().clamp_min(1.0),
            "teacher_helpful_rate": (
                (teacher_gate >= 0.5).to(valid_mask.dtype) * valid_mask
            ).sum()
            / valid_mask.sum().clamp_min(1.0),
            **{name: value.mean() for name, value in diagnostics.items()},
        }

    def ranking_nce_loss(
        self, query: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        logits = self.loss_weight * (query @ target.t())
        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)
