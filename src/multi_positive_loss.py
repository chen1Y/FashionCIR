"""Positive-aware losses shared by ranking, the gate teacher and hard mining."""

import torch
import torch.nn.functional as F


def positive_mask(logits, mask=None):
    if mask is None:
        mask = torch.eye(*logits.shape, device=logits.device, dtype=torch.bool)
    if mask.shape != logits.shape or mask.dtype != torch.bool:
        raise ValueError("Positive mask must be boolean with the same shape as logits")
    mask = mask.to(logits.device)
    if not mask.any(dim=1).all():
        raise ValueError("Each query must have at least one positive")
    return mask


def multi_positive_nce(logits, mask=None, reduction="mean"):
    """Mean log-probability of ALL labelled positives (SupCon-style).

    With a diagonal mask this is ordinary one-way cross entropy. Computing
    log-softmax in float32 avoids half-precision overflow for large logits.
    """
    if mask is None and logits.shape[0] == logits.shape[1]:
        return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device),
                               reduction=reduction)
    mask = positive_mask(logits, mask)
    log_prob = F.log_softmax(logits.float(), dim=1)
    loss = -log_prob.masked_fill(~mask, 0).sum(1) / mask.sum(1)
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unknown reduction: {reduction}")


def positive_aware_margin(baseline, adapted, mask, valid, k, margin):
    """Mine only true negatives; compare mean positive similarities.

    Rows with fewer than k negatives use only the available ones. All-positive
    batches return a differentiable zero, with no inf/NaN subtraction.
    """
    mask = positive_mask(baseline, mask)
    if k <= 0:
        zero = adapted.sum() * 0
        return zero, zero.detach()
    k = min(k, baseline.shape[1])
    with torch.no_grad():
        indices = baseline.masked_fill(mask, -torch.inf).topk(k, dim=1).indices
        available = (~mask).gather(1, indices)
        base_positive = baseline.masked_fill(~mask, 0).sum(1) / mask.sum(1)
        base_gap = base_positive[:, None] - baseline.gather(1, indices)
    adapted_positive = adapted.masked_fill(~mask, 0).sum(1) / mask.sum(1)
    adapted_gap = adapted_positive[:, None] - adapted.gather(1, indices)
    violations = F.relu(float(margin) + base_gap - adapted_gap)
    per_row = violations.masked_fill(~available, 0).sum(1) / available.sum(1).clamp_min(1)
    row_valid = valid.reshape(-1).float() * available.any(1)
    loss = (per_row * row_valid).sum() / row_valid.sum().clamp_min(1)
    weights = available * row_valid[:, None]
    rate = ((violations > 0) * weights).sum() / weights.sum().clamp_min(1)
    return loss, rate
