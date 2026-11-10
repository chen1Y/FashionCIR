"""Stable candidate rank evaluation, including the Torch2.9 keyword-only overload."""
import torch


def candidate_ranks(scores:torch.Tensor,mask:torch.Tensor,original_order:torch.Tensor)->torch.Tensor:
    if scores.shape!=mask.shape or mask.shape!=original_order.shape or not mask[:,0].all():raise ValueError('Invalid candidate bindings')
    if not torch.isfinite(scores[mask]).all():raise FloatingPointError('Nonfinite real candidate scores')
    score=scores.masked_fill(~mask,-torch.inf)
    original=torch.argsort(original_order,dim=-1,stable=True)
    ranking=original.gather(1,torch.argsort(score.gather(1,original),dim=-1,descending=True,stable=True))
    return (ranking==0).nonzero(as_tuple=False)[:,1]+1
