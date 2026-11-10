"""Feature-space visual replacement for the 029D written reference."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from fiq_fusion_029 import fuse_parts, fuse_text, objective


class VisualConflictEraserV2(nn.Module):
    def __init__(self, initial_gamma=0.5):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(float(initial_gamma)))

    def forward(self, reference, drop, active):
        direction = (reference * drop).sum(-1, keepdim=True).relu() * drop
        residual = reference - self.gamma.clamp(0., 1.5) * direction
        norm = residual.norm(dim=-1, keepdim=True)
        return torch.where(active[:, None] & (norm > 1e-6), residual / norm.clamp_min(1e-6), reference)


class LatentVisualProxyMLP(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.residual = nn.Sequential(nn.Linear(4 * dimension, dimension), nn.LayerNorm(dimension),
                                      nn.GELU(), nn.Dropout(0.1), nn.Linear(dimension, dimension))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.blend_logit = nn.Parameter(torch.tensor(math.log(2.0)))

    def forward(self, clean, text, drop, active):
        shift = F.normalize(text - drop, dim=-1, eps=1e-6) * active[:, None]
        residual = self.residual(torch.cat((clean, text, clean * text, shift), dim=-1)).float()
        proxy = F.normalize(clean + residual, dim=-1)
        alpha = 0.75 * self.blend_logit.sigmoid()
        final = F.normalize((1. - alpha) * clean + alpha * proxy, dim=-1)
        return final, proxy


class PixelFreeVisual(nn.Module):
    def __init__(self, dimension, use_conflict_eraser, use_latent_proxy):
        super().__init__()
        self.eraser = VisualConflictEraserV2() if use_conflict_eraser else None
        self.proxy = LatentVisualProxyMLP(dimension) if use_latent_proxy else None

    def forward(self, reference, text, drop, active):
        reference, text, drop = reference.float(), text.float(), drop.float()
        clean = self.eraser(reference, drop, active) if self.eraser is not None else reference
        return self.proxy(clean, text, drop, active) if self.proxy is not None else (clean, None)


def compose(model, visual, parts, reference, drop, active, written, arm):
    if visual is None:
        return fuse_parts(model, parts, reference, written, arm)
    text, beta, slots_weight = fuse_text(model, parts, reference, arm)
    visual_feature, proxy = visual(reference, text, drop, active)
    query, alpha = model.image_fusion(text, visual_feature)
    return dict(q=query, text=text, alpha=alpha, beta=beta, slot_weights=slots_weight,
                global_text=parts['eos'], slots=parts['slots'], valid=parts['valid'], proxy=proxy)


def visual_losses(proxy, query, target, tau=0.07):
    target = target.detach().float()
    rec = (1. - (proxy.float() * target).sum(-1)).mean()
    teacher = (target @ target.T / tau).softmax(dim=1)
    student = (query.float() @ target.T / tau).log_softmax(dim=1)
    kl = F.kl_div(student, teacher, reduction='batchmean')
    return rec, kl


def loss_for(out, gallery, labels, mask, teacher, reference, arm, *, lambda_rec, lambda_kl,
             reference_weight):
    losses = objective(out, gallery, labels, mask, teacher, arm, reference=reference,
                       reference_weight=reference_weight)
    if out.get('proxy') is not None:
        with torch.autocast(device_type=out['q'].device.type, enabled=False):
            rec, kl = visual_losses(out['proxy'], out['q'], gallery[labels])
        losses.update(rec=rec, kl=kl, total=losses['total'] + lambda_rec * rec + lambda_kl * kl)
    return losses
