"""Post-hoc branch-specific controls; no fitting or hyperparameter search."""


def transition_counts(reference, candidate, k):
    if len(reference) != len(candidate) or not reference or k < 1:
        raise ValueError('Inconsistent rank vectors or threshold')
    if any(not isinstance(r,int) or r < 1 for r in reference+candidate):
        raise ValueError('Ranks must be positive integers')
    gains=[];losses=[];both=neither=0
    for i,(old,new) in enumerate(zip(reference,candidate)):
        before,after=old<=k,new<=k
        if after and not before:gains.append(i)
        elif before and not after:losses.append(i)
        elif before and after:both+=1
        else:neither+=1
    return dict(queries=len(reference),k=k,gains=len(gains),losses=len(losses),both_hit=both,neither_hit=neither,
        net_queries=len(gains)-len(losses),delta_pp=100*(len(gains)-len(losses))/len(reference),
        gained_query_indices=gains,lost_query_indices=losses)


def branch_controls(frozen_logit, low_rank, valid, mask=None):
    import torch
    from fiq_candidate_head import shuffled_correction
    if frozen_logit.shape != low_rank.shape or frozen_logit.ndim != 2:
        raise ValueError('Inconsistent branch tensor shapes')
    if mask is None:mask=torch.ones_like(low_rank,dtype=torch.bool)
    frozen_logit=torch.where(mask,frozen_logit,torch.zeros_like(frozen_logit))
    low_rank=torch.where(mask,low_rank,torch.zeros_like(low_rank))
    if not torch.isfinite(frozen_logit).all() or not torch.isfinite(low_rank).all():
        raise FloatingPointError('Nonfinite real control input')
    def shuffle(value):
        if bool(mask.all()):return shuffled_correction(value)
        rng=torch.Generator().manual_seed(42);result=torch.zeros_like(value)
        for j,row in enumerate(value):
            positions=mask[j].nonzero(as_tuple=False).flatten()
            result[j,positions]=row[positions[torch.randperm(len(positions),generator=rng)]]
        return result
    active=mask & valid[:,None].bool()
    zero=torch.where(active,.03*torch.tanh(frozen_logit),torch.zeros_like(low_rank))
    paired=torch.where(active,.03*torch.tanh(frozen_logit+low_rank),torch.zeros_like(low_rank))
    branch_only=torch.where(active,.03*torch.tanh(frozen_logit+shuffle(low_rank)),torch.zeros_like(low_rank))
    return dict(zero_rank4=zero,paired_rank4=paired,whole_correction_shuffle=shuffle(paired),rank4_only_shuffle=branch_only)
