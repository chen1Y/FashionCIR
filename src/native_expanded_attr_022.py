"""Actual batch8/full-bank optimizer smoke; one step is discarded, never a fit."""
import argparse,json
from pathlib import Path
import torch
from train_expanded_attr import load_context
from expanded_attr_assets import base_path
from expanded_attr_model import PairAlignedText,branch_forward,compose,losses
from train_attr_slot_pilot import load_clip
from generate_structured_edits import atomic_write_json


def smoke(root,cat,arm):
    torch.manual_seed(42);torch.set_num_threads(4);c=load_context(root,cat,'train');clip,_,tok=load_clip(base_path(cat));model=PairAlignedText(clip,tok,2).cuda();model.eval();optimizer=torch.optim.AdamW(model.parameter_groups());b=c['banks']['fit'];local=torch.arange(8,device='cuda');ix=torch.tensor(b['indices'][:8],device='cuda');old=model.embedding.weight[:49408].detach().clone()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out=branch_forward(model,[c['rows'][i] for i in ix.tolist()],arm);q=compose(out,c['reference'][ix],arm);ls=losses(out,q,b['gallery'],b['labels'][local],b['mask'][local],c['teacher'][ix],arm)
    if not all(torch.isfinite(v).all() for v in ls.values()):raise FloatingPointError('Nonfinite native loss')
    ls['total'].backward();gradient=model.embedding.weight.grad
    if gradient[:49408].count_nonzero() or not torch.isfinite(gradient).all() or any(p.grad is not None for p in clip.visual.parameters()):raise ValueError('Batch8 freeze/hook failed')
    norms=gradient[-4:].norm(dim=1).tolist();params=[p for g in optimizer.param_groups for p in g['params']];norm=torch.nn.utils.clip_grad_norm_(params,1.)
    if not torch.isfinite(norm):raise FloatingPointError('Native gradient nonfinite')
    optimizer.step()
    if not torch.equal(old,model.embedding.weight[:49408]):raise ValueError('Old vocabulary changed')
    atomic_write_json(root/('native_'+cat+'_'+arm+'.json'),dict(status='passed',batch=8,bank_size=len(b['ids']),old_vocabulary_exact=True,vit_gradients=False,new_row_gradient_norms=norms,one_smoke_update_discarded=True,loss={k:float(v.detach()) for k,v in ls.items()},max_cuda_gib=torch.cuda.max_memory_allocated()/2**30))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--category',required=True);p.add_argument('--arm',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();smoke(a.root,a.category,a.arm)
