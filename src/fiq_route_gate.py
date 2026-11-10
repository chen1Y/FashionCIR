"""A bounded query-only route gate and image-disjoint train/dev sampling."""
import hashlib

import torch
from torch import nn
from torch.nn import functional as F


def sample_partitions(rows, excluded, seed=42, train_limit=1024, dev_limit=256):
    def bucket(image_id):
        return int(hashlib.sha256(f'{seed}|{image_id}'.encode()).hexdigest()[:8],16)%4
    fit,dev = [],[]
    dropped = dict(official_heldout=0,cross_partition=0)
    for row in rows:
        left,right = row['candidate'],row['target']
        if left in excluded or right in excluded:
            dropped['official_heldout'] += 1
        elif (bucket(left)==0) != (bucket(right)==0):
            dropped['cross_partition'] += 1
        else:
            (dev if bucket(left)==0 else fit).append(row)
    def key(row):
        return hashlib.sha256(f"{seed}|{row['candidate']}|{row['target']}|{row['source_index']}".encode()).hexdigest()
    counts = dict(available_fit=len(fit),available_dev=len(dev),**dropped)
    fit,dev = sorted(fit,key=key)[:train_limit],sorted(dev,key=key)[:dev_limit]
    ids = lambda rows:{r[k] for r in rows for k in ('candidate','target')}
    if ids(fit)&ids(dev) or (ids(fit)|ids(dev))&set(excluded):
        raise ValueError('Gate split image leakage')
    return fit,dev,counts


def features(qd,qq,confidence,valid):
    agreement = (qd*qq).sum(-1).clamp(-1,1)
    distance = (qd-qq).norm(dim=-1).clamp(0,2)
    return torch.stack([confidence,valid.float(),agreement,distance],dim=-1).float()


class RouteGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4,16),nn.GELU(),nn.Linear(16,1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self,x,valid):
        alpha = .5+.25*torch.tanh(self.network(x).squeeze(-1))
        return torch.where(valid.bool(),alpha,torch.zeros_like(alpha))


def mixed_query(qd,qq,alpha):
    result = F.normalize((1-alpha[:,None])*qd+alpha[:,None]*qq,dim=-1)
    return torch.where((alpha==0)[:,None],qd,result)


def shuffled_valid_alpha(alpha,valid,seed=42):
    indices = valid.nonzero(as_tuple=False).flatten()
    order = torch.randperm(len(indices),generator=torch.Generator().manual_seed(seed))
    result = alpha.clone()
    result[indices] = alpha[indices[order]]
    return result


def contrastive_loss(query,targets,source_ids,target_ids,temperature=.07):
    scores = query@targets.T/temperature
    positive = torch.tensor([[a==b for b in target_ids] for a in target_ids],device=query.device)
    reference = torch.tensor([[a==b for b in target_ids] for a in source_ids],device=query.device)
    scores = scores.masked_fill(reference & ~positive,-torch.inf)
    # Exact repeated target IDs are positives, not false in-batch negatives.
    return (torch.logsumexp(scores,-1)-torch.logsumexp(scores.masked_fill(~positive,-torch.inf),-1)).mean()
