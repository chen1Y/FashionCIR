"""020 matched text branches, full legal banks and R10/R50 checkpoint selection."""
import json
import torch
from torch.nn import functional as F
from attr_token_pair_alignment import PairAlignedText
from train_clean_attr_alignment import plain_text

ARMS=('original','slots','dual')

def branch_forward(model,records,arm):
 if arm not in ARMS:raise ValueError('Unregistered arm')
 mod=None
 if arm!='slots':mod=plain_text(model,[r['modification_text'] for r in records])
 if arm=='original':return dict(q=None,t_mod=mod,global_text=mod,slots=None,valid=None)
 out=model([r['target_attrs'] for r in records]);valid=out['attr_valid'];any_valid=valid.any(1)
 average=torch.where(valid[...,None],out['f_attr_text'],torch.zeros_like(out['f_attr_text'])).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
 generated=F.normalize(out['f_global_text']+average,dim=-1)
 if mod is None and not any_valid.all():
  # Fallback rows still get real original-text gradients, not fake empty slots.
  mod=torch.zeros_like(generated);idx=(~any_valid).nonzero().flatten();mod[idx]=plain_text(model,[records[i]['modification_text'] for i in idx.tolist()])
 if arm=='slots':text=torch.where(any_valid[:,None],generated,mod if mod is not None else torch.zeros_like(generated))
 else:text=torch.where(any_valid[:,None],.5*mod+.5*generated,mod)
 return dict(q=None,t_mod=mod,text=text,global_text=out['f_global_text'],slots=out['f_attr_text'],valid=valid)

def compose(out,reference,arm):return F.normalize(reference.float()+(out['t_mod'] if arm=='original' else out['text']),dim=-1)

def legal_masks(metadata,split):
 ids=metadata['banks'][split];hashes=metadata['image_sha256'];index={i:j for j,i in enumerate(ids)};canonical=set(metadata['canonical'][split]);base=torch.tensor([i in canonical for i in ids],dtype=torch.bool)
 rows=[metadata['records'][i] for i in metadata['offsets'][split]];masks=[];labels=[]
 for r in rows:
  mask=base.clone();gt=index[r['target']];key=json.dumps((hashes[r['candidate']],r['modification_text']))
  forbidden=set(metadata['known_positive_sha'].get(key,[]))|{hashes[r['candidate']],hashes[r['target']]}
  for j,image_id in enumerate(ids):
   if hashes[image_id] in forbidden:mask[j]=False
  mask[gt]=True
  if mask.sum()<2:raise ValueError('No legal negatives')
  negatives=[hashes[ids[j]] for j in mask.nonzero().flatten().tolist() if j!=gt]
  if len(negatives)!=len(set(negatives)) or set(negatives)&forbidden:raise ValueError('Negative byte collision')
  masks.append(mask);labels.append(gt)
 return torch.stack(masks),torch.tensor(labels,dtype=torch.long)

def losses(out,q,gallery,labels,mask,teacher,arm):
 scores=q@gallery.T;ret=F.cross_entropy(scores.masked_fill(~mask,-torch.inf)/.07,labels);target=gallery[labels]
 if arm=='original':global_loss=(1-(out['global_text']*target).sum(-1)).mean();slot=ret*0.
 else:
  valid=out['valid'];good=valid.any(1);dist=1-(out['global_text']*target).sum(-1);global_loss=torch.where(good,dist,torch.zeros_like(dist)).sum()/good.sum().clamp_min(1)
  slot_distance=1-(out['slots']*teacher).sum(-1);slot=torch.where(valid,slot_distance,torch.zeros_like(slot_distance)).sum()/valid.sum().clamp_min(1)
 return dict(total=ret+.1*global_loss+.1*slot,retrieval=ret,global_alignment=global_loss,slot_alignment=slot)

def ranks_from_scores(scores,labels,references):
 scores=scores.clone();batch=torch.arange(len(labels),device=scores.device)
 if (labels==references).any():raise ValueError('Reference is GT, original ID evaluation undefined')
 scores[batch,references]=-torch.inf;target=scores[batch,labels]
 if not torch.isfinite(target).all():raise FloatingPointError('Nonfinite GT scores')
 tie_before=torch.arange(scores.shape[1],device=scores.device)[None]<labels[:,None]
 return 1+(scores>target[:,None]).sum(1)+((scores==target[:,None])&tie_before).sum(1)

def metrics(ranks):return dict(queries=len(ranks),ranks=ranks,**{'r'+str(k):100*sum(r<=k for r in ranks)/len(ranks) for k in (1,10,50)})

def selection_key(metric,nce):return ((metric['r10']+metric['r50'])/2,-nce)
