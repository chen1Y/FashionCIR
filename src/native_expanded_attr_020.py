"""Native020 architecture smoke on fixed prior real pairs; no persistent fit."""
import argparse,json,copy
from pathlib import Path
import torch
from expanded_attr_model import PairAlignedText,branch_forward,compose,losses,ARMS
from expanded_attr_assets import base_path
from expanded_attr_metadata import OLD
from train_attr_slot_pilot import load_clip
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.manual_seed(42);torch.set_num_threads(4)
 path=Path('/dev/shm/fiq_attr_clean_019_shirt.pt');expected=json.loads((OLD/'attr_clean_alignment_019/shirt/results.json').read_text())
 if file_sha(path)!=expected['new_cache_sha256']:raise ValueError('Real native preflight source changed')
 c=torch.load(path,map_location='cpu',weights_only=True);rows=[dict(modification_text=r['view']['mod_text'],target_attrs=r['view']['target_attrs']) for r in c['records'][:2]]
 clip,_,tok=load_clip(base_path('shirt'));initial=PairAlignedText(clip,tok,2).delta_state();report={}
 for arm in ARMS:
  m=PairAlignedText(clip,tok,2).cuda();m.load_delta(initial);m.eval();clip.zero_grad(set_to_none=True)
  optimizer=torch.optim.AdamW(m.parameter_groups());old=m.embedding.weight[:49408].detach().clone()
  with torch.autocast('cuda',dtype=torch.bfloat16):
   o=branch_forward(m,rows,arm);q=compose(o,c['v_ref'][:2].cuda(),arm);d=losses(o,q,c['targets'][:2,0].cuda(),torch.arange(2,device='cuda'),torch.ones(2,2,dtype=torch.bool,device='cuda'),c['slot_teacher'][:2].cuda(),arm)
  d['total'].backward();g=m.embedding.weight.grad
  if g[:49408].count_nonzero() or any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None) or any(p.grad is not None for p in clip.visual.parameters()):raise ValueError('Native020 gradient failed')
  norm=g[-4:].norm(dim=1).tolist()
  if arm!='original' and not all(v>0 for v in norm):raise ValueError('New token has no gradient')
  torch.nn.utils.clip_grad_norm_([p for g_ in optimizer.param_groups for p in g_['params']],1.);optimizer.step()
  if not torch.equal(old,m.embedding.weight[:49408]):raise ValueError('Old vocabulary drift')
  report[arm]=dict(queries=list(q.shape),new_token_gradient_norms=norm,old_row_gradient_nonzeros=0,old_rows_exact_after_step=True,vit_gradients=False,smoke_updates=1,weights_discarded=True,loss={k:float(v.detach()) for k,v in d.items()})
 atomic_write_json(a.root/'native_preflight.json',dict(status='passed',arms=report,scope='Architecture smoke uses first2 prior019 real pairs; not020 training/results.'))
 print(json.dumps({'status':'passed','arms':report}),flush=True)

if __name__=='__main__':main()
