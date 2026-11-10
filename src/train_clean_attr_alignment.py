"""019 clean-slot counterfactual using018 geometry, optimizer settings and pairs."""
import argparse,copy,gc,json,time,hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
import train_attr_pair_alignment as previous
from attr_token_pair_alignment import PairAlignedText,paired_losses,pair_query,packed_audit
from attr_slot_data import SLOTS
from attr_slot_rank_math import candidate_ranks
from experiment_storage import require_space,atomic_torch_save
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json

def plain_text(model,texts):
 tokens=model.tokenizer(texts).to(model.embedding.weight.device);clip=model.clip
 x=model.embedding(tokens)+clip.positional_embedding
 x=clip.transformer(x.permute(1,0,2),attn_mask=clip.attn_mask).permute(1,0,2)
 x=clip.ln_final(x)@clip.text_projection
 eos=(tokens==model.tokenizer.eot_token_id).int().argmax(-1)
 return F.normalize(x[torch.arange(len(tokens),device=x.device),eos].float(),dim=-1)

def forward(model,cache,ix):
 attrs=[cache['records'][i]['view']['target_attrs'] for i in ix.tolist()];o=model(attrs)
 mod=cache['mod'][ix]
 missing=~o['attr_valid'].any(1)
 if missing.any():
  mod=mod.clone();indices=missing.nonzero().flatten();texts=[cache['records'][int(ix[i])]['view']['mod_text'] for i in indices.tolist()]
  mod[indices]=plain_text(model,texts)
 return o,mod

def loss_for(out,reference,mod,candidates,mask,teacher):
 d=paired_losses(out,reference,mod,candidates,mask,teacher)
 # All original queries remain supervised. Rejected generations use original
 # modification text (last2 still train); never impose invented empty-slot GT.
 valid=out['attr_valid'].any(1);global_loss=(1-(out['f_global_text']*candidates[:,0]).sum(-1))
 d['global_alignment']=torch.where(valid,global_loss,torch.zeros_like(global_loss)).sum()/valid.sum().clamp_min(1)
 d['total']=d['retrieval']+.1*d['global_alignment']+.1*d['slot_alignment']
 return d

def evaluate(model,c,indices,baseline=False):
 ranks=[];sums={k:0. for k in ('retrieval','global_alignment','slot_alignment')}
 with torch.no_grad():
  for j in range(0,len(indices),4):
   ix=torch.tensor(indices[j:j+4],device='cuda')
   with torch.autocast('cuda',dtype=torch.bfloat16):
    if baseline:q=F.normalize(c['v_ref'][ix]+c['mod'][ix],dim=-1)
    else:
     out,mod=forward(model,c,ix);q=pair_query(out,c['v_ref'][ix],mod);ls=loss_for(out,c['v_ref'][ix],mod,c['targets'][ix],c['mask'][ix],c['slot_teacher'][ix])
     for k in sums:sums[k]+=float(ls[k])*len(ix)
   score=torch.einsum('bd,bmd->bm',q,c['targets'][ix]);ranks+=candidate_ranks(score,c['mask'][ix],c['original_order'][ix]).tolist()
   if baseline:sums['retrieval']+=float(F.cross_entropy(score.masked_fill(~c['mask'][ix],-torch.inf)/.07,torch.zeros(len(ix),dtype=torch.long,device='cuda')))*len(ix)
 return dict(queries=len(indices),ranks=ranks,**{'candidate_pool_r'+str(k):100*sum(x<=k for x in ranks)/len(ranks) for k in (1,5,10)},**{k:v/len(indices) for k,v in sums.items()})

def token_stats(model,initial):
 now=model.embedding.weight[-4:].detach().float();orig=initial['new_token_rows'].to(now)
 return {slot:dict(initial_norm=float(orig[k].norm()),current_norm=float(now[k].norm()),delta_norm=float((now[k]-orig[k]).norm()),relative_change_percent=100*float((now[k]-orig[k]).norm()/orig[k].norm().clamp_min(1e-12)),cosine_to_initial=float(F.cosine_similarity(now[k:k+1],orig[k:k+1]))) for k,slot in enumerate(SLOTS)}

def snapshot(model,c,indices):
 result=[]
 with torch.no_grad():
  for i in indices:
   ix=torch.tensor([i],device='cuda')
   with torch.autocast('cuda',dtype=torch.bfloat16):out,mod=forward(model,c,ix);q=pair_query(out,c['v_ref'][ix],mod)
   score=torch.einsum('bd,bmd->bm',q,c['targets'][ix]);rank=int(candidate_ranks(score,c['mask'][ix],c['original_order'][ix])[0]);r=c['records'][i];order=sorted(range(len(r['pool_ids'])),key=lambda k:(-float(score[0,k]),r['original_order'][k]))
   result.append(dict(index=i,split=r['split'],source_index=r['source_index'],gt_rank=rank,gt_cosine=float(score[0,0]),top3=[dict(image_id=r['pool_ids'][k],cosine=float(score[0,k]),is_gt=k==0) for k in order[:3]],attribute_vectors=out['f_attr_text'][0].cpu().tolist(),valid=out['attr_valid'][0].tolist()))
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--research',type=Path,required=True);p.add_argument('--dataset',choices=('shirt','toptee'),required=True);a=p.parse_args();r=a.research;cat=a.dataset;start=time.monotonic();torch.set_num_threads(4);torch.manual_seed(42)
 output=r/'attr_clean_alignment_019'/cat
 if output.exists():raise FileExistsError('Preserve019 output; no blind restart')
 require_space(r);weights=Path('/data/8T1/lcy/CIRData/runs/fiq-small-weights/attr_clean_019')/cat
 if weights.exists():raise FileExistsError('Existing019 weights require inspection')
 require_space('/home/lcy',min_free_gib=2);output.mkdir(parents=True);weights.mkdir(parents=True)
 generation=r/'clean_attr_generation_019';completion=json.loads((generation/'completion.json').read_text())
 if completion['status']!='complete':raise ValueError('Generation not complete')
 oldmanifest=json.loads((r/'attr_pair_alignment_018'/cat/'manifest.json').read_text());oldresults=json.loads((r/'attr_pair_alignment_018'/cat/'results.json').read_text());cache_path=Path('/dev/shm/fiq_attr_pair_018_'+cat+'.pt')
 if file_sha(cache_path)!=oldresults['cache_sha256']:raise ValueError('Original018 geometry cache changed')
 c=torch.load(cache_path,map_location='cpu',weights_only=True)
 if c['records']!=oldmanifest['records']:raise ValueError('Records/source changed')
 raw_records=copy.deepcopy(c['records']);generated=json.loads((generation/'generated_slots.json').read_text())['categories'][cat]
 if len(generated)!=len(raw_records):raise ValueError('No filtering of failed generations')
 for rec,g in zip(c['records'],generated):
  if any(rec[k]!=g[k] for k in ('original_row','source_index','split','reference','target')) or rec['view']['mod_text']!=g['modification_text']:raise ValueError('Generation/query binding changed')
  rec['view']=dict(target_attrs=g['target_attrs'],mod_text=g['modification_text'],drop_attrs='',trace=g['errors'],fallback=g['status']!='ok')
 base=Path('/data/8T1/lcy/CIRData/autodl-tmp/results/fiq_shirt_toptee_20261022')/cat/'checkpoints'/(cat+'_dqu_original_seed42_best.pt')
 if file_sha(base)!=c['evidence']['base_sha256']:raise ValueError('Protected checkpoint changed')
 clip,transform,tokenizer=previous.old.load_clip(base)
 #018 saved global geometry/soft teachers, but NOT target patch tensors. Extract
 #only these missing target patches once; NEVER reencode gallery/ref globals.
 target_ids=list(dict.fromkeys(x['target'] for x in c['records']));patches={};images=Path('/data/8T1/lcy/CIRData/autodl-tmp/data/FashionIQ/resized_image');clip.visual.output_tokens=True
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  for j in range(0,len(target_ids),8):
   ids=target_ids[j:j+8];ims=[]
   for image_id in ids:
    path=images/(image_id+'.jpg')
    if file_sha(path)!=oldmanifest['image_sha256'][image_id]:raise ValueError('Target JPEG bytes changed')
    with Image.open(path) as im:ims.append(transform(im.convert('RGB')))
   _,p_=clip.visual(torch.stack(ims).cuda())
   if clip.visual.final_ln_after_pool:p_=clip.visual.ln_post(p_)
   p_=F.normalize((p_@clip.visual.proj).float(),dim=-1).half().cpu()
   patches.update({image_id:p_[k] for k,image_id in enumerate(ids)})
  clip.visual.output_tokens=False
  texts=[row['view']['target_attrs'][s] or 'unspecified' for row in c['records'] for s in SLOTS];features=[]
  for j in range(0,len(texts),16):features.append(F.normalize(clip.encode_text(tokenizer(texts[j:j+16]).cuda()).float(),dim=-1).cpu())
  attr=torch.cat(features).reshape(len(c['records']),4,-1)
 teacher=[]
 for i,row in enumerate(c['records']):
  p_=patches[row['target']].float();teacher.append(F.normalize(torch.softmax(attr[i]@p_.T/.07,dim=-1)@p_,dim=-1))
 c['slot_teacher']=torch.stack(teacher);c['evidence019']=dict(generation_sha256=file_sha(generation/'generated_slots.json'),old_geometry_sha256=file_sha(cache_path),target_patch_images=len(target_ids))
 atomic_torch_save(dict(ids=target_ids,patches=patches),Path('/dev/shm/fiq_attr_target_patches_019_'+cat+'.pt'))
 newcache=Path('/dev/shm/fiq_attr_clean_019_'+cat+'.pt');atomic_torch_save(c,newcache);cache_sha=file_sha(newcache)
 c={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in c.items()};offset=oldmanifest['offsets'];model=PairAlignedText(clip,tokenizer,2).cuda();model.eval()
 packed=packed_audit(model,[row['view']['target_attrs'] for row in c['records']])
 if any(any(x['truncated']) for x in packed):raise ValueError('Accepted/fallback clean text must fit actual77-token packer')
 atomic_write_json(output/'packed_inputs.json',dict(records=packed,source_indices=[x['source_index'] for x in c['records']]));atomic_write_json(output/'manifest.json',dict(records=c['records'],offsets=offset,evidence=c['evidence019']))
 protocol=dict(status='sealed',dataset=cat,seed=42,fit=128,dev=32,steps=50,batch=4,token_lr=.0002,last_two_lr=.00001,embedding_weight_decay=0.,block_weight_decay=.01,selection='dev retrieval NCE at0/25/50',original_control_results_sha256=file_sha(r/'attr_pair_alignment_018'/cat/'results.json'),new_cache_sha256=cache_sha,evidence=c['evidence019'],api_generation_cost=completion['cost'],old_controls_reused_without_retraining=True,all_rows_retained=True,official_validation_used=False,weight_directory=str(weights))
 atomic_write_json(output/'protocol.json',protocol)
 baseline=evaluate(None,c,offset['dev'],True)
 if baseline!=oldresults['baseline']:raise ValueError('Frozen baseline did not reproduce018')
 initial=model.delta_state();best=copy.deepcopy(initial);initial_metrics=evaluate(model,c,offset['dev']);best_loss=initial_metrics['retrieval'];best_step=0
 history=[dict(step=0,metrics=initial_metrics,tokens=token_stats(model,initial))];fixed=offset['fit'][:2]+offset['dev'][:2];first=snapshot(model,c,fixed);frozen=previous.frozen_digest(clip)
 optimizer=torch.optim.AdamW(model.parameter_groups());params=[p for g in optimizer.param_groups for p in g['params']];rng=torch.Generator().manual_seed(42);exposure=torch.zeros(128,dtype=torch.long);first_grad=None
 for step in range(1,51):
  require_space(r)
  if (r/'attr_clean_019_control/cancel.request').exists():raise KeyboardInterrupt('User stopped019')
  if time.monotonic()-start>3300:raise TimeoutError('019 category exceeds1h')
  local=torch.randperm(128,generator=rng)[:4];exposure[local]+=1;ix=local.cuda();optimizer.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):out,mod=forward(model,c,ix);loss=loss_for(out,c['v_ref'][ix],mod,c['targets'][ix],c['mask'][ix],c['slot_teacher'][ix])
  if not torch.isfinite(loss['total']):raise FloatingPointError('Nonfinite loss')
  loss['total'].backward();grad=model.embedding.weight.grad
  if grad[:model.embedding.base_size].count_nonzero() or any(p.grad is not None for p in clip.parameters() if not p.requires_grad):raise ValueError('Frozen gradient leak')
  if first_grad is None:first_grad=grad[-4:].norm(dim=1).tolist()
  norm=torch.nn.utils.clip_grad_norm_(params,1.)
  if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
  optimizer.step()
  if not torch.equal(model.embedding.weight[:model.embedding.base_size],clip.token_embedding.weight):raise ValueError('Old vocabulary drift')
  if step in (25,50):
   metrics=evaluate(model,c,offset['dev']);history.append(dict(step=step,metrics=metrics,tokens=token_stats(model,initial),train={k:float(v.detach()) for k,v in loss.items()}));atomic_write_json(output/'progress.json',history)
   if metrics['retrieval']<best_loss:best_loss=metrics['retrieval'];best_step=step;best=model.delta_state()
 if previous.frozen_digest(clip)!=frozen:raise ValueError('Frozen weights changed')
 if exposure.tolist()!=oldresults['arms']['tokens_last2']['exposure']:raise ValueError('Exposure differs from018 control')
 model.load_delta(best);selected=evaluate(model,c,offset['dev']);last=snapshot(model,c,fixed);stats=token_stats(model,initial)
 atomic_torch_save(dict(state=best,selected_step=best_step,protocol=protocol),weights/'clean_last2_delta.pt')
 traces=[]
 for before,after in zip(first,last):
  i=before['index'];row=c['records'][i];v0=torch.tensor(before.pop('attribute_vectors'));v1=torch.tensor(after.pop('attribute_vectors'));cos=F.cosine_similarity(v0,v1,dim=-1).tolist()
  traces.append(dict(index=i,split=row['split'],source_index=row['source_index'],reference=row['reference'],target=row['target'],modification_text=row['view']['mod_text'],old_attrs=raw_records[i]['view']['target_attrs'],new_attrs=row['view']['target_attrs'],actual_input=packed[i]['actual_packed_text'],fallback=row['view']['fallback'],times_in_training=int(exposure[i]) if i<128 else 0,before=before,after=after,attribute_cosine_to_initial=dict(zip(SLOTS,cos))))
 atomic_write_json(output/'process_examples.json',dict(selection='Fixed first2fit+first2dev, no rank-based selection',examples=traces,shared_token_stats=stats))
 result=dict(status='complete',category=cat,baseline=baseline,original018=oldresults['arms']['tokens_last2']['selected'],initial=initial_metrics,selected=selected,selected_step=best_step,history=history,tokens=stats,first_token_gradient_norms=first_grad,updates=50,presentations=int(exposure.sum()),exposure=exposure.tolist(),frozen_sha256_before=frozen,frozen_sha256_after=frozen,old_vocabulary_exact=True,vit_frozen=True,generated_valid=sum(not x['view']['fallback'] for x in c['records']),queries_retained=len(c['records']),all_rows_retained=True,weight_path=str(weights/'clean_last2_delta.pt'),weight_sha256=file_sha(weights/'clean_last2_delta.pt'),new_cache_sha256=cache_sha,official_val_queries_loaded=0,full_gallery_evaluated=False,elapsed_seconds=time.monotonic()-start)
 atomic_write_json(output/'results.json',result);print(json.dumps(dict(status='complete',category=cat,selected_step=best_step,selected=selected)),flush=True)

if __name__=='__main__':main()
