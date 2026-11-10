"""A bounded candidate-level head; frozen input statistics and exact head-only ranks."""
import torch
from torch import nn
from fiq_small_math import blend


def candidate_features(qd,qq,reference,candidates,confidence):
    left=torch.einsum('bd,bkd->bk',qd,candidates)
    right=torch.einsum('bd,bkd->bk',qq,candidates)
    retain=torch.einsum('bd,bkd->bk',reference,candidates)
    agreement=(qd*qq).sum(-1).clamp(-1,1)[:,None].expand_as(left)
    result=torch.stack([left,right,retain,right-left,agreement,confidence[:,None].expand_as(left)],-1)
    if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite candidate features')
    return result.float()


def fit_statistics(x):
    flat=x.reshape(-1,6)
    return flat.mean(0),flat.std(0,unbiased=False).clamp_min(1e-3)


class CandidateHead(nn.Module):
    def __init__(self,mean,std):
        super().__init__()
        self.register_buffer('mean',mean.detach().clone())
        self.register_buffer('std',std.detach().clone())
        self.network=nn.Sequential(nn.Linear(6,16),nn.GELU(),nn.Linear(16,1))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self,x,valid):
        correction=.03*torch.tanh(self.network((x-self.mean)/self.std).squeeze(-1))
        return torch.where(valid[:,None].bool(),correction,torch.zeros_like(correction))


def training_pool(qd,qq,gallery,sources,targets,allowed,rows,gallery_ids):
    scores=blend(qd,qq,.5)@gallery.T
    negative=scores.clone();negative[:,~allowed]=-torch.inf
    positives={}
    for row in rows:
        positives.setdefault((row['candidate'],row['modification_text']),set()).add(row['target'])
    index={image_id:i for i,image_id in enumerate(gallery_ids)}
    for i,row in enumerate(rows):
        if not bool(allowed[sources[i]]) or not bool(allowed[targets[i]]):
            raise ValueError('Supervised source/target outside its image partition')
        negative[i,sources[i]]=-torch.inf
        for image_id in positives[(row['candidate'],row['modification_text'])]:
            negative[i,index[image_id]]=-torch.inf
    hard=torch.argsort(negative,descending=True,stable=True)[:,:31]
    if not torch.isfinite(negative.gather(1,hard)).all():raise ValueError('Fewer than31 eligible hard negatives')
    pool=torch.cat([targets[:,None],hard],1)
    if (pool==sources[:,None]).any() or (hard==targets[:,None]).any():
        raise ValueError('Reference/known positive incorrectly used as negative')
    return pool,scores.gather(1,pool)


def shuffled_correction(correction):
    rng=torch.Generator().manual_seed(42)
    return torch.stack([row[torch.randperm(len(row),generator=rng)] for row in correction])


def head_ranks(order,targets,adjusted_head_scores):
    baseline=(order==targets[:,None]).nonzero(as_tuple=False)[:,1]+1
    head=order[:,:adjusted_head_scores.shape[1]]
    # The caller supplies the corrected score of each head item; tie order is retained.
    local=torch.argsort(adjusted_head_scores,dim=-1,descending=True,stable=True)
    new_head=head.gather(1,local)
    matches=(new_head==targets[:,None]).nonzero(as_tuple=False)
    result=baseline.clone()
    result[matches[:,0]]=matches[:,1]+1
    return result
