"""020 full eligible-bank training and complete official-ID retrieval evaluation."""
import argparse,copy,json,time,math
from pathlib import Path
import torch
from expanded_attr_metadata import OLD,ARMS
from expanded_attr_assets import CACHE,base_path
from expanded_attr_model import PairAlignedText,branch_forward,compose,legal_masks,losses,ranks_from_scores,metrics,selection_key
from train_attr_slot_pilot import load_clip
from train_attr_pair_alignment import frozen_digest
from experiment_storage import require_space,atomic_torch_save
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json

def guard(root,start):
 require_space(OLD);require_space(root,2)
 if (root/'cancel.request').exists():raise KeyboardInterrupt('User stopped020')
 if time.monotonic()-start>3300:raise TimeoutError('020 small job exceeds3300s')

def load_context(root,cat,phase):
 m=json.loads((root/'metadata'/(cat+'_'+phase+'.json')).read_text());p=CACHE/(cat+'_'+phase+'_images.pt');d=torch.load(p,map_location='cpu',weights_only=True)
 report=json.loads((root/('assets_'+cat+'_'+phase+'.json')).read_text())
 if file_sha(p)!=report['cache_sha256'] or file_sha(root/'metadata'/(cat+'_'+phase+'.json'))!=d['metadata_sha256']:raise ValueError('Image bank binding changed')
 if file_sha(base_path(cat))!=d['base_sha256']:raise ValueError('Protected category checkpoint changed')
 generated=json.loads((root/'texts'/(cat+'_'+phase+'.json')).read_text())['samples']
 if len(generated)!=len(m['records']):raise ValueError('No query dropping')
 rows=[]
 for r,g in zip(m['records'],generated):
  if r['source_index']!=g['source_index'] or r['candidate']!=g['reference'] or r['modification_text']!=g['instruction']:raise ValueError('Online query binding changed')
  rows.append(dict(r,target_attrs=g['target_attrs'],generation_status=g['status'],generation_errors=g['errors']))
 index={i:j for j,i in enumerate(d['ids'])};allgallery=d['gallery'];ref=torch.stack([allgallery[index[r['candidate']]] for r in rows]).cuda()
 context=dict(metadata=m,rows=rows,reference=ref,banks={},image_cache_sha256=report['cache_sha256'],text_sha256=file_sha(root/'texts'/(cat+'_'+phase+'.json')))
 if phase=='train':
  teacherpath=CACHE/(cat+'_teacher.pt');teacher=torch.load(teacherpath,map_location='cpu',weights_only=True)
  if file_sha(teacherpath)!=json.loads((root/('teacher_'+cat+'.json')).read_text())['cache_sha256']:raise ValueError('Teacher cache changed')
  if teacher['text_sha256']!=context['text_sha256'] or teacher['image_cache_sha256']!=context['image_cache_sha256']:raise ValueError('Teacher version mismatch')
  context['teacher']=teacher['teachers'].cuda()
  for split in ('fit','dev'):
   ids=m['banks'][split];bankindex={i:j for j,i in enumerate(ids)};indices=m['offsets'][split];mask,labels=legal_masks(m,split)
   context['banks'][split]=dict(ids=ids,gallery=torch.stack([allgallery[index[i]] for i in ids]).cuda(),indices=indices,
    labels=labels.cuda(),references=torch.tensor([bankindex[rows[i]['candidate']] for i in indices],device='cuda'),mask=mask.cuda())
 else:
  ids=m['gallery_ids'];context['banks']['val']=dict(ids=ids,gallery=allgallery.cuda(),indices=list(range(len(rows))),labels=torch.tensor([index[r['target']] for r in rows],device='cuda'),references=torch.tensor([index[r['candidate']] for r in rows],device='cuda'))
 return context

def evaluate(model,context,split,arm):
 b=context['banks'][split];ranks=[];nce=[]
 with torch.no_grad():
  for j in range(0,len(b['indices']),8):
   selected=b['indices'][j:j+8];ix=torch.tensor(selected,device='cuda');rows=[context['rows'][i] for i in selected]
   with torch.autocast('cuda',dtype=torch.bfloat16):out=branch_forward(model,rows,arm);q=compose(out,context['reference'][ix],arm)
   score=q@b['gallery'].T;labels=b['labels'][j:j+len(selected)];references=b['references'][j:j+len(selected)]
   ranks+=ranks_from_scores(score,labels,references).tolist()
   if split!='val':logits=score.masked_fill(~b['mask'][j:j+len(selected)],-torch.inf)/.07
   else:
    logits=score.clone();logits[torch.arange(len(selected),device='cuda'),references]=-torch.inf;logits=logits/.07
   nce+=torch.nn.functional.cross_entropy(logits,labels,reduction='none').tolist()
 return dict(**metrics(ranks),retrieval_nce=sum(nce)/len(nce))

def fit(root,cat,arm,smoke=False):
 start=time.monotonic();torch.manual_seed(42);torch.set_num_threads(4);guard(root,start);c=load_context(root,cat,'train');clip,_,tok=load_clip(base_path(cat));model=PairAlignedText(clip,tok,2).cuda();model.eval();optimizer=torch.optim.AdamW(model.parameter_groups());params=[p for g in optimizer.param_groups for p in g['params']]
 b=c['banks']['fit'];n=len(b['indices']);exposure=torch.zeros(n,dtype=torch.long);rng=torch.Generator().manual_seed(42)
 if smoke:
  local=torch.arange(min(2,n),device='cuda');ix=torch.tensor(b['indices'][:len(local)],device='cuda');old=model.embedding.weight[:49408].detach().clone()
  with torch.autocast('cuda',dtype=torch.bfloat16):out=branch_forward(model,[c['rows'][i] for i in ix.tolist()],arm);q=compose(out,c['reference'][ix],arm);ls=losses(out,q,b['gallery'],b['labels'][local],b['mask'][local],c['teacher'][ix],arm)
  ls['total'].backward();gradient=model.embedding.weight.grad
  if gradient[:49408].count_nonzero() or not torch.isfinite(gradient).all() or any(p.grad is not None for p in clip.visual.parameters()):raise ValueError('Native freeze/hook check failed')
  torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
  if not torch.equal(old,model.embedding.weight[:49408]):raise ValueError('Old vocabulary drift')
  atomic_write_json(root/('native_'+cat+'_'+arm+'.json'),dict(status='passed',optimizer_smoke_updates=1,weights_discarded=True,old_vocabulary_exact=True,vit_gradients=False,new_row_gradient_norms=gradient[-4:].norm(dim=1).tolist(),bank_size=len(b['ids']),loss={k:float(v.detach()) for k,v in ls.items()},max_cuda_gib=torch.cuda.max_memory_allocated()/2**30));return
 output=root/'fits'/cat/arm
 if output.exists():raise FileExistsError('Never blindly restart020 fit')
 output.mkdir(parents=True);frozen=frozen_digest(clip);history=[];best=None;selected_epoch=0;steps=0
 def record(epoch):
  nonlocal best,selected_epoch
  value=evaluate(model,c,'dev',arm);key=selection_key(value,value['retrieval_nce']);path=output/('epoch_'+str(epoch)+'_delta.pt')
  atomic_torch_save(dict(state=model.delta_state(),epoch=epoch,arm=arm,category=cat,image_cache_sha256=c['image_cache_sha256'],text_sha256=c['text_sha256']),path)
  history.append(dict(epoch=epoch,steps=steps,metrics=value,selection_score=key[0],weight_path=str(path),weight_sha256=file_sha(path)))
  if best is None or key>best:best=key;selected_epoch=epoch
  atomic_write_json(output/'progress.json',dict(status='training',history=history,selected_epoch=selected_epoch,exposure=exposure.tolist(),updates=steps))
 record(0)
 for epoch in range(1,4):
  order=torch.randperm(n,generator=rng)
  for j in range(0,n,8):
   guard(root,start);local=order[j:j+8];exposure[local]+=1;gpu=local.cuda();ix=torch.tensor([b['indices'][i] for i in local.tolist()],device='cuda');optimizer.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):out=branch_forward(model,[c['rows'][i] for i in ix.tolist()],arm);q=compose(out,c['reference'][ix],arm);ls=losses(out,q,b['gallery'],b['labels'][gpu],b['mask'][gpu],c['teacher'][ix],arm)
   if not torch.isfinite(ls['total']):raise FloatingPointError('Nonfinite020 loss')
   ls['total'].backward()
   if model.embedding.weight.grad[:49408].count_nonzero() or any(p.grad is not None for p in clip.parameters() if not p.requires_grad):raise ValueError('Frozen gradient leak')
   norm=torch.nn.utils.clip_grad_norm_(params,1.)
   if not torch.isfinite(norm):raise FloatingPointError('Nonfinite020 gradient')
   optimizer.step();steps+=1
   if not torch.equal(model.embedding.weight[:49408],clip.token_embedding.weight):raise ValueError('Old embedding changed')
   if steps%16==0:
    with (output/'training.jsonl').open('a') as log:log.write(json.dumps(dict(epoch=epoch,step=steps,presentations=int(exposure.sum()),loss={k:float(v.detach()) for k,v in ls.items()},gradient_norm=float(norm)))+'\n')
  record(epoch)
 if not (exposure==3).all() or frozen_digest(clip)!=frozen:raise ValueError('Exposure/freeze integrity')
 chosen=history[selected_epoch]
 result=dict(status='complete',category=cat,arm=arm,history=history,selected_epoch=selected_epoch,selected_weight=chosen['weight_path'],selected_weight_sha256=chosen['weight_sha256'],updates=steps,sample_presentations=int(exposure.sum()),per_fit_row_exposures=exposure.tolist(),fit_queries=n,dev_queries=len(c['banks']['dev']['indices']),fit_gallery=len(b['ids']),dev_gallery=len(c['banks']['dev']['ids']),frozen_sha256_before=frozen,frozen_sha256_after=frozen,old_vocabulary_exact=True,selection='max mean(devR10,devR50), then min devNCE, then earlier epoch',official_validation_loaded=False,image_cache_sha256=c['image_cache_sha256'],text_sha256=c['text_sha256'],seconds=time.monotonic()-start,max_cuda_gib=torch.cuda.max_memory_allocated()/2**30)
 atomic_write_json(output/'results.json',result);atomic_write_json(output/'fit_complete.json',dict(status='complete',selected_weight_sha256=chosen['weight_sha256'],official_validation_loaded=False));print(json.dumps({'status':'complete','cat':cat,'arm':arm,'epoch':selected_epoch,'dev':chosen['metrics']}),flush=True)

def official(root,cat):
 for category in ('shirt','toptee'):
  for arm in ARMS:
   if not (root/'fits'/category/arm/'fit_complete.json').exists():raise ValueError('Cannot evaluate until every fit is sealed')
 output=root/'official'/cat
 if output.exists():raise FileExistsError('Official validation reports once per frozen020 protocol')
 output.mkdir(parents=True);start=time.monotonic();guard(root,start);torch.set_num_threads(4);c=load_context(root,cat,'val');clip,_,tok=load_clip(base_path(cat));model=PairAlignedText(clip,tok,2).cuda();model.eval()
 rows={'frozen_original':evaluate(model,c,'val','original')}
 for arm in ARMS:
  guard(root,start);fitresult=json.loads((root/'fits'/cat/arm/'results.json').read_text());weight=Path(fitresult['selected_weight'])
  if file_sha(weight)!=fitresult['selected_weight_sha256']:raise ValueError('Selected checkpoint changed')
  model.load_delta(torch.load(weight,map_location='cpu',weights_only=True)['state']);rows[arm]=evaluate(model,c,'val',arm)
 atomic_write_json(output/'results.json',dict(status='complete',category=cat,queries=len(c['rows']),gallery=len(c['banks']['val']['ids']),rows=rows,reference_id_excluded=True,gt_inserted=False,all_official_queries_retained=True,image_cache_sha256=c['image_cache_sha256'],text_sha256=c['text_sha256'],seconds=time.monotonic()-start));print(json.dumps({'status':'complete','category':cat,'metrics':{n:{k:v for k,v in d.items() if k!='ranks'} for n,d in rows.items()}}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--category',choices=('shirt','toptee'),required=True);p.add_argument('--arm',choices=ARMS);p.add_argument('--smoke',action='store_true');p.add_argument('--official',action='store_true');a=p.parse_args()
 if a.official:official(a.root,a.category)
 else:fit(a.root,a.category,a.arm,smoke=a.smoke)
