"""Conflict-neutralized main text and modification-conditioned slot gating."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from fiq_fusion_029 import TextMixtureGate


class ModificationSlotGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, slots, eos, reference, modification, valid):
        s, e, v, m = (item.float() for item in (slots, eos, reference, modification))
        identity = torch.eye(4, device=s.device).expand(len(s), -1, -1)
        fields = ((s * v[:, None]).sum(-1, keepdim=True), (s * e[:, None]).sum(-1, keepdim=True),
                  (v * e).sum(-1)[:, None, None].expand(-1, 4, -1), identity,
                  (s * m[:, None]).sum(-1, keepdim=True))
        scores = self.mlp(torch.cat(fields, dim=-1)).squeeze(-1).float()
        active = valid.any(-1, keepdim=True)
        safe = torch.where(active, scores.masked_fill(~valid, -torch.inf), torch.zeros_like(scores))
        return safe.softmax(dim=-1) * valid


class ResidualTextGate(TextMixtureGate):
    def __init__(self):
        super().__init__()
        # 0.45 * sigmoid(-log(2)) = 0.15.
        nn.init.constant_(self.network[-1].bias, -math.log(2))

    def forward(self, modification, generated, reference, valid):
        fields = torch.stack(((modification * generated).sum(-1), (modification * reference).sum(-1),
                              (generated * reference).sum(-1), valid.float().mean(-1),
                              valid.any(-1).float()), dim=-1)
        beta = .45 * self.network(fields).float().sigmoid().squeeze(-1)
        return beta * valid.any(-1)


def ordinary_text(model, texts):
    """Use frozen original rows for ordinary tokens, preserving last-two-layer gradients."""
    clip = model.clip
    tokens = model.tokenizer(texts).to(clip.token_embedding.weight.device)
    hidden = clip.token_embedding(tokens) + clip.positional_embedding
    hidden = clip.transformer(hidden.permute(1, 0, 2), attn_mask=clip.attn_mask).permute(1, 0, 2)
    projected = clip.ln_final(hidden) @ clip.text_projection
    eos = (tokens == model.tokenizer.eot_token_id).int().argmax(-1)
    return F.normalize(projected[torch.arange(len(tokens), device=tokens.device), eos].float(), dim=-1)


def compose_query(model, visual, rows, reference, drop, drop_active):
    main, mod = ordinary_text(model, [r['main_context_text'] for r in rows] +
                              [r['mod_clean'] for r in rows]).chunk(2)
    attr = model([r['target_attrs'] for r in rows])
    eos, slots, valid = attr['f_global_text'], attr['f_attr_text'], attr['attr_valid']
    weights = model.attribute_gate(slots, eos, reference, mod, valid)
    u = F.normalize(eos + (slots * weights[..., None]).sum(1), dim=-1)
    beta = model.text_mixture(mod, u, reference, valid)
    mixed = F.normalize((1. - beta[:, None]) * main + beta[:, None] * u, dim=-1)
    visual_feature, _ = visual(reference, mixed, drop, drop_active)
    q, alpha = model.image_fusion(mixed, visual_feature)
    return dict(q=q, text=mixed, alpha=alpha, beta=beta, slot_weights=weights,
                global_text=eos, slots=slots, valid=valid)


def conflict_opr(model, rows, target):
    active = [i for i, row in enumerate(rows) if row['drop_attrs']]
    if not active:
        return target.new_zeros(())
    d = ordinary_text(model, [rows[i]['drop_attrs'] for i in active])
    return F.relu((d * target[active].detach().float()).sum(-1)).square().mean()
