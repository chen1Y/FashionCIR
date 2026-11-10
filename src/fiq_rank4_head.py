"""A same-start scalar control and a zero-initialized rank4 vector interaction."""
import math
import torch
from torch import nn


def pack_vectors(scalar, query, candidates, mask=None):
    if scalar.shape[-1] != 6 or query.shape[-1] != candidates.shape[-1]:
        raise ValueError('Unexpected scalar/vector feature dimensions')
    value = torch.cat((scalar, query[:,None,:].expand(-1,scalar.shape[1],-1), candidates), -1)
    if mask is not None:
        value = torch.where(mask[...,None], value, torch.zeros_like(value))
    if not torch.isfinite(value).all():
        raise FloatingPointError('Nonfinite real packed vector feature')
    return value


class Rank4Head(nn.Module):
    def __init__(self, scalar, dimension, arm):
        super().__init__()
        if arm not in ('scalar_continue', 'rank4_branch'):
            raise ValueError('Unsupported fixed012 arm')
        self.scalar = scalar
        self.dimension = dimension
        self.arm = arm
        if arm == 'rank4_branch':
            for p in self.scalar.parameters():
                p.requires_grad_(False)
            self.query_projection = nn.Linear(dimension,4,bias=False)
            self.candidate_projection = nn.Linear(dimension,4,bias=False)
            nn.init.zeros_(self.candidate_projection.weight)

    def forward(self, x, valid):
        if x.shape[-1] != 6+2*self.dimension:
            raise ValueError('Packed vector dimension mismatch')
        scalar = x[...,:6]
        if self.arm == 'scalar_continue':
            return self.scalar(scalar,valid)
        q = x[...,6:6+self.dimension]
        candidate = x[...,6+self.dimension:]
        frozen_logit = self.scalar.network((scalar-self.scalar.mean)/self.scalar.std).squeeze(-1)
        low_rank = (self.query_projection(q)*self.candidate_projection(candidate)).sum(-1)/math.sqrt(4)
        correction = .03*torch.tanh(frozen_logit+low_rank)
        return torch.where(valid[:,None].bool(),correction,torch.zeros_like(correction))
