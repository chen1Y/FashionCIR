"""029D slot gate with modification cosine, lower EOS weight and strict spans."""

import torch
from torch import nn
from torch.nn import functional as F


LOGIT_SCALE = 5.0
EOS_WEIGHT = 0.2


class ModificationAwareGate(nn.Module):
    def __init__(self, original_gate):
        super().__init__()
        first, _, last = original_gate.mlp
        self.mlp = nn.Sequential(nn.Linear(8, first.out_features), nn.GELU(),
                                 nn.Linear(first.out_features, 1))
        with torch.no_grad():
            self.mlp[0].weight[:, :7].copy_(first.weight)
            self.mlp[0].weight[:, 7].zero_()
            self.mlp[0].bias.copy_(first.bias)
            self.mlp[2].load_state_dict(last.state_dict())

    def forward(self, slots, eos, reference, valid, modification):
        slots, eos, reference = slots.float(), eos.float(), reference.float()
        identity = torch.eye(4, device=slots.device).expand(len(slots), -1, -1)
        features = torch.cat([
            (slots * reference[:, None]).sum(-1, keepdim=True),
            (slots * eos[:, None]).sum(-1, keepdim=True),
            (reference * eos).sum(-1)[:, None, None].expand(-1, 4, -1),
            identity,
            F.cosine_similarity(slots, modification.float()[:, None], dim=-1)[..., None],
        ], dim=-1)
        logits = LOGIT_SCALE * self.mlp(features).squeeze(-1).float()
        safe = torch.where(valid.any(-1, keepdim=True),
                           logits.masked_fill(~valid, -torch.inf), torch.zeros_like(logits))
        return safe.softmax(-1) * valid


def compose_refreshed(model, parts, reference, written, arm):
    if arm != 'full_learned_written':
        raise ValueError('Slot refresh is defined for the full 029D arm')
    mod, eos, slots, valid = (parts[key] for key in ('mod', 'eos', 'slots', 'valid'))
    weights = model.attribute_gate(slots, eos, reference, valid, mod)
    generated = F.normalize(EOS_WEIGHT * eos + (weights[..., None] * slots).sum(1), dim=-1)
    beta = model.text_mixture(mod, generated, reference, valid)
    text = F.normalize((1 - beta[:, None]) * mod + beta[:, None] * generated, dim=-1)
    query, alpha = model.image_fusion(text, written)
    return dict(q=query, text=text, alpha=alpha, beta=beta, slot_weights=weights,
                global_text=eos, slots=slots, valid=valid)
