"""Small frozen-feature hypotheses; no labels are used to form query scores."""
import torch
import torch.nn.functional as F


def blend(left, right, alpha):
    weight = torch.as_tensor(alpha, dtype=left.dtype, device=left.device)
    if weight.ndim == 0:
        if weight.item() == 0:
            return left
        if weight.item() == 1:
            return right
    if weight.ndim == 1:
        weight = weight[:, None]
    mixed = F.normalize((1-weight)*left+weight*right, dim=-1)
    return torch.where(weight == 0, left, torch.where(weight == 1, right, mixed))


def adaptive_alpha(confidence, valid):
    # A fixed, bounded query-only hypothesis, not a learned uncertainty theorem.
    return torch.where(valid.bool(), .5 + .1*(2*confidence.clamp(0,1)-1),
                       torch.zeros_like(confidence))


def ranks_from_scores(scores, targets):
    order = torch.argsort(scores, dim=-1, descending=True)
    return (order == targets[:, None]).nonzero(as_tuple=False)[:, 1] + 1


def block_rerank(scores, attribute_scores, targets, weight=.03, k=100):
    """Only permute the first k candidates; preserve original ties and the tail."""
    if weight == 0:
        return ranks_from_scores(scores, targets)
    order = torch.argsort(scores, dim=-1, descending=True)
    k = min(k, scores.shape[1])
    head = order[:, :k].clone()
    adjusted = scores.gather(1,head) + weight*attribute_scores.gather(1,head)
    local = torch.argsort(adjusted, dim=-1, descending=True, stable=True)
    order[:, :k] = head.gather(1,local)
    return (order == targets[:,None]).nonzero(as_tuple=False)[:,1]+1


def metrics(ranks):
    return {**{f"r{k}":float((ranks <= k).float().mean()*100) for k in (1,10,50,100)},
            "mean_rank":float(ranks.float().mean()), "mrr":float((1/ranks.float()).mean())}
