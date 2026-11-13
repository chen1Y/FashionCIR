"""Last-two-block ViT LoRA and the live-positive gallery objective."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from torch.utils.checkpoint import checkpoint


class QVLoRA(nn.Module):
    """Apply low-rank updates to Q and V in OpenCLIP's packed QKV projection."""

    def __init__(self, width, rank, alpha, device):
        super().__init__()
        self.a = nn.Parameter(torch.empty(2, rank, width, device=device))
        self.b = nn.Parameter(torch.zeros(2, width, rank, device=device))
        for matrix in self.a:
            nn.init.kaiming_uniform_(matrix, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, weight):
        q, k, v = weight.chunk(3, dim=0)
        delta_q, delta_v = (self.scale * (self.b @ self.a)).to(weight.dtype).unbind()
        return torch.cat((q + delta_q, k, v + delta_v), dim=0)


def attach_lora(visual, rank=8, alpha=16):
    for block in list(visual.transformer.resblocks)[-2:]:
        weight = block.attn.in_proj_weight
        width = weight.shape[1]
        if weight.shape != (3 * width, width) or weight.requires_grad:
            raise ValueError('039 requires a frozen, packed QKV projection')
        adapter = QVLoRA(width, rank, alpha, weight.device)
        parametrize.register_parametrization(block.attn, 'in_proj_weight', adapter)
        yield adapter


def tail_features(visual, hidden, *, checkpoint_blocks=False):
    """Continue the ViT from the output of block N-3, using its normal pooling."""
    if visual.attn_pool is not None:
        raise ValueError('039 expects ViT-H/14 without attention pooling')
    x = hidden.transpose(0, 1)
    for block in list(visual.transformer.resblocks)[-2:]:
        x = checkpoint(block, x, use_reentrant=False) if checkpoint_blocks else block(x)
    x = x.transpose(0, 1)
    if visual.final_ln_after_pool:
        pooled, _ = visual._global_pool(x)
        pooled = visual.ln_post(pooled)
    else:
        pooled, _ = visual._global_pool(visual.ln_post(x))
    if visual.proj is not None:
        pooled = pooled @ visual.proj
    return F.normalize(pooled.float(), dim=-1)


def replace_positive(scores, query, positive, labels):
    live_scores = (query * positive).sum(-1, keepdim=True).to(scores.dtype)
    return scores.scatter(1, labels[:, None], live_scores)


def live_gallery_loss(out, gallery, labels, mask, teacher, positive):
    """Only each query's actual positive is live. Other gallery entries are detached."""
    query = out['q']
    gallery = gallery.detach()
    logits = replace_positive(query @ gallery.T, query, positive, labels)
    retrieval = F.cross_entropy(logits.masked_fill(~mask, -torch.inf) / .07, labels)
    valid = out['valid']
    good = valid.any(1)
    global_dist = 1 - (out['global_text'] * positive.detach()).sum(-1)
    glob = torch.where(good, global_dist, 0.).sum() / good.sum().clamp_min(1)
    slot_dist = 1 - (out['slots'] * teacher).sum(-1)
    slot = torch.where(valid, slot_dist, 0.).sum() / valid.sum().clamp_min(1)
    with torch.autocast(device_type=query.device.type, enabled=False):
        bank = gallery.detach().float()
        target = positive.detach().float()
        teacher_scores = replace_positive(target @ bank.T, target, target, labels)
        teacher_probs = (teacher_scores / .07).masked_fill(~mask, -torch.inf).softmax(dim=1)
        student_scores = replace_positive(query.float() @ bank.T, query.float(), positive.float(), labels)
        student_log = (student_scores / .07).masked_fill(~mask, -torch.inf).log_softmax(dim=1)
        kl = F.kl_div(student_log.masked_fill(~mask, 0.), teacher_probs, reduction='batchmean')
    return dict(total=retrieval + .1 * glob + .1 * slot + .5 * kl, retrieval=retrieval,
                global_alignment=glob, slot_alignment=slot, gallery_kl=kl)
