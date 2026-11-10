"""017 real OpenCLIP attribute-slot feasibility pilot on fixed byte-isolated pools."""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
import open_clip
from attr_slot_data import slot_view,SLOTS
from attr_slot_fusion import AttrSlotCIR
from experiment_storage import require_space,atomic_torch_save
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json

ARMS={'full':(True,True,True),'no_block':(False,True,True),'no_eraser':(True,False,True),'no_ortho':(True,True,False)}


def metadata(a):
    r=a.research;cat=a.dataset;f=r/'supported_candidate_head_011'/cat
    s=json.loads((f.parent/'completion_summary.json').read_text())['categories'][cat]
    if file_sha(f/'manifest.json')!=s['manifest_sha256']:raise ValueError('011 manifest changed')
    m=json.loads((f/'manifest.json').read_text());poolsha=file_sha(f/'candidate_pools.pt');p016=r/'visible_competition_016'/cat/'results.json';d016=json.loads(p016.read_text())
    if d016['status']!='complete' or poolsha!=d016['evidence']['pool011_sha256'] or file_sha(f/'manifest.json')!=d016['evidence']['manifest011_sha256']:raise ValueError('016 proof does not match011')
    gm=r/'train_gallery_diagnostic'/cat/'manifest.json'
    if file_sha(gm)!=d016['evidence']['gallery005_manifest_sha256']:raise ValueError('005 bytes changed')
    hashes=json.loads(gm.read_text())['image_sha256'];p=a.primary/cat/f'structured_{cat}_train.json';expected=json.loads((r/'train_only_route_gate'/cat/'protocol.json').read_text())['evidence']
    if file_sha(p)!=expected['structured_train_sha256']:raise ValueError('Qwen source changed')
    generated={x['source_index']:x for x in json.loads(p.read_text())['samples']};allrows=m['original_fit']+m['original_dev'];records=[];offsets={};actual={}
    for split,n in [('fit',128),('dev',32)]:
        selected=m['selected_original_rows'][split][:n];pools=m['candidate_ids'][split][:n];details={x['original_query_row']:x for x in d016['splits'][split]['queries']};offsets[split]=list(range(len(records),len(records)+n));actual[split]=set()
        if len(selected)!=n:raise ValueError('Do not relax128/32 minimum')
        for original,pool in zip(selected,pools):
            row=allrows[original];g=generated[row['source_index']]
            if any(g[k]!=row[k] for k in ('candidate','target','modification_text')) or details[original]['pool_ids']!=pool:raise ValueError('Query/pool binding changed')
            view=slot_view(g,row['modification_text']);head=[x['image_id'] for x in details[original]['candidates']]
            if view['fallback'] or row['target']!=pool[0] or len(pool)<2:raise ValueError('Invalid inherited support')
            records.append(dict(split=split,original_row=original,source_index=row['source_index'],reference=row['candidate'],target=row['target'],pool_ids=pool,
                original_order=[head.index(i) for i in pool],saved_fixed05_visible_rank=details[original]['visible_gt_rank'],view=view))
            actual[split].update(hashes[i] for i in pool+[row['candidate']])
    if actual['fit']&actual['dev']:raise ValueError('Learning-input bytes overlap')
    base=a.primary/cat/'checkpoints'/f'{cat}_dqu_original_seed42_best.pt'
    if file_sha(base)!=expected['base_sha256']:raise ValueError('Protected DQU checkpoint changed')
    evidence=dict(category=cat,task_document_sha256=a.task_sha256,base_sha256=expected['base_sha256'],manifest011_sha256=file_sha(f/'manifest.json'),
        pool011_sha256=poolsha,source016_sha256=file_sha(p016),structured_train_sha256=file_sha(p),gallery005_manifest_sha256=file_sha(gm),
        source_hashes={name:file_sha(Path(__file__).with_name(name)) for name in ('attr_slot_data.py','attr_slot_fusion.py','train_attr_slot_pilot.py')},
        open_clip_version=importlib.metadata.version('open_clip_torch'),torch_version=str(torch.__version__),precision='FP32 protected weights; BF16 autocast; normalized FP32 scores',
        fit_queries=128,dev_queries=32,steps=50,batch_size=4,seed=42,learning_rate=.0002,weight_decay=.01,temperature=.07,lambda_ortho=.01,lambda_erase=.01,clip_gradient=1.)
    return records,offsets,hashes,base,evidence


def load_clip(base):
    clip,_,transform=open_clip.create_model_and_transforms('ViT-H-14',pretrained=None)
    checkpoint=torch.load(base,map_location='cpu',weights_only=True,mmap=True);state=checkpoint['model_state']
    clip.load_state_dict({k[5:]:v for k,v in state.items() if k.startswith('clip.')},strict=True)
    del state,checkpoint;clip.requires_grad_(False);clip.eval();clip=clip.cuda();return clip,transform,open_clip.get_tokenizer('ViT-H-14')


def encode_cache(a,clip,transform,tokenizer,records,hashes,evidence):
    refs=list(dict.fromkeys(r['reference'] for r in records));images=list(dict.fromkeys(i for r in records for i in r['pool_ids']));globals={};patches={};checked={}
    def batch(ids):
        tensors=[]
        for image_id in ids:
            path=a.root/'resized_image'/(image_id+'.jpg');digest=file_sha(path)
            if digest!=hashes[image_id]:raise ValueError('Image byte version changed')
            checked[image_id]=digest
            with Image.open(path) as im:tensors.append(transform(im.convert('RGB')))
        return torch.stack(tensors).cuda()
    visual=clip.visual
    if visual.attn_pool is not None or visual.proj is None:raise ValueError('Pilot requires native OpenCLIP ViT token projection; EVA adapter not assumed')
    start=time.monotonic()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        visual.output_tokens=True
        for j in range(0,len(refs),8):
            ids=refs[j:j+8];global_,tokens=visual(batch(ids))
            if visual.final_ln_after_pool:tokens=visual.ln_post(tokens)
            projected=tokens@visual.proj
            for k,i in enumerate(ids):globals[i]=F.normalize(global_[k].float(),dim=-1).cpu();patches[i]=projected[k].half().cpu()
        visual.output_tokens=False
        remaining=[i for i in images if i not in globals]
        for j in range(0,len(remaining),8):
            ids=remaining[j:j+8];v=F.normalize(clip.encode_image(batch(ids)).float(),dim=-1).cpu()
            for k,i in enumerate(ids):globals[i]=v[k]
        def texts(values):
            output=[]
            for j in range(0,len(values),16):output.append(F.normalize(clip.encode_text(tokenizer(values[j:j+16]).cuda()).float(),dim=-1).cpu())
            return torch.cat(output)
        mod=texts([r['view']['mod_text'] for r in records]);drop=torch.zeros_like(mod);has=[i for i,r in enumerate(records) if r['view']['drop_attrs']]
        if has:drop[has]=texts([records[i]['view']['drop_attrs'] for i in has])
    pool=torch.zeros(len(records),20,1024);mask=torch.zeros(len(records),20,dtype=torch.bool);order=torch.full((len(records),20),99,dtype=torch.long)
    for j,r in enumerate(records):
        n=len(r['pool_ids']);pool[j,:n]=torch.stack([globals[i] for i in r['pool_ids']]);mask[j,:n]=True;order[j,:n]=torch.tensor(r['original_order'])
    cache=dict(evidence=evidence,image_sha256=checked,records=records,patches=torch.stack([patches[r['reference']] for r in records]),
        v_cls=torch.stack([globals[r['reference']] for r in records]),t_mod=mod,t_drop=drop,targets=pool,mask=mask,original_order=order,
        encoding_seconds=time.monotonic()-start,new_reference_patch_images=len(refs),global_candidate_images=len(images),total_unique_images=len(checked))
    for key in ('patches','v_cls','t_mod','t_drop','targets'):
        if not torch.isfinite(cache[key]).all():raise FloatingPointError('Nonfinite new pilot cache')
    atomic_torch_save(cache,a.cache);return cache


def evaluate(model,cache,indices,baseline=False):
    ranks=[];losses=[];model.eval() if model is not None else None
    with torch.no_grad():
        for start in range(0,len(indices),4):
            ix=torch.tensor(indices[start:start+4],device='cuda');records=[cache['records'][i] for i in ix.tolist()]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                q=F.normalize(cache['v_cls'][ix]+cache['t_mod'][ix],dim=-1) if baseline else forward(model,cache,ix)['q_final']
            scores=torch.einsum('bd,bmd->bm',q.float(),cache['targets'][ix]);scores=scores.masked_fill(~cache['mask'][ix],-torch.inf)
            # Stable ties use the original frozen-head order, not positive storage column0.
            original=cache['original_order'][ix].argsort(-1,stable=True);sorted_=original.gather(1,scores.gather(1,original).argsort(-1,descending=True,stable=True));ranks.extend((sorted_==0).nonzero(as_tuple=False)[:,1].add(1).tolist())
            losses.extend(F.cross_entropy(scores/.07,torch.zeros(len(ix),device='cuda',dtype=torch.long),reduction='none').tolist())
    return dict(queries=len(indices),candidate_pool_r1=100*sum(r<=1 for r in ranks)/len(ranks),candidate_pool_r5=100*sum(r<=5 for r in ranks)/len(ranks),candidate_pool_r10=100*sum(r<=10 for r in ranks)/len(ranks),retrieval_nce=sum(losses)/len(losses),ranks=ranks)


def forward(model,cache,ix):
    rows=[cache['records'][i]['view'] for i in ix.tolist()];has=torch.tensor([bool(r['drop_attrs']) for r in rows],device='cuda')
    return model([r['target_attrs'] for r in rows],cache['patches'][ix],cache['v_cls'][ix],cache['t_mod'][ix],cache['t_drop'][ix],has)


def dev_objective(model,cache,indices):
    values=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(indices),4):
            ix=torch.tensor(indices[start:start+4],device='cuda');has=torch.tensor([bool(cache['records'][i]['view']['drop_attrs']) for i in ix.tolist()],device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                o=forward(model,cache,ix);loss=model.losses(o['q_final'],cache['targets'][ix],o['f_attr_text'],cache['t_drop'][ix],has,o['attr_valid'],cache['mask'][ix])
            values.append(float(loss['retrieval']))
    return sum(values)/len(values)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=('shirt','toptee'),required=True)
    for n in ('root','primary','research','output','cache'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--task-sha256',required=True);p.add_argument('--prepare-cache',action='store_true');a=p.parse_args()
    started=time.monotonic();torch.set_num_threads(4);torch.manual_seed(42)
    if a.output.exists():raise FileExistsError('Preserve pilot, no blind rerun')
    require_space(a.research,min_free_gib=8.12)
    if not torch.cuda.is_available():raise RuntimeError('Native pilot needs GPU')
    records,offsets,hashes,base,evidence=metadata(a);clip,transform,tokenizer=load_clip(base)
    if a.prepare_cache:
        if a.cache.exists():raise FileExistsError('No duplicate new-pilot encoding')
        cache=encode_cache(a,clip,transform,tokenizer,records,hashes,evidence)
    else:
        if not a.cache.exists():raise ValueError('Prepared cache missing; do not silently reencode')
        cache=torch.load(a.cache,map_location='cpu',weights_only=True)
        if cache['evidence']!=evidence or cache['records']!=records:raise ValueError('Prepared cache provenance changed')
    cache={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in cache.items()};cache['patches']=cache['patches'].float()
    if a.prepare_cache:
        model=AttrSlotCIR(clip,tokenizer).cuda();ix=torch.tensor(offsets['fit'][:4],device='cuda');model.train()
        with torch.autocast('cuda',dtype=torch.bfloat16):o=forward(model,cache,ix);loss=o['q_final'].square().mul(torch.randn_like(o['q_final'])).sum()
        loss.backward();gradient=model.text.embedding.extra.grad.norm(dim=1)
        if not torch.isfinite(gradient).all() or not (gradient>0).all() or any(p.grad is not None for p in clip.parameters()):raise ValueError('Native new-token gradient/frozen-base check failed')
        if model.text.tokenizer.encode('<|align_color|>')!=[model.text.special_ids['<|align_color|>']]:raise ValueError('Native tokenizer registration failed')
        report=dict(status='native_preflight_passed',evidence=evidence,cache_sha256=file_sha(a.cache),actual_output_shape=list(o['f_attr_text'].shape),new_token_gradient_norms=gradient.tolist(),base_gradients=False,patch_shape=list(cache['patches'].shape),new_reference_patch_images=cache['new_reference_patch_images'],global_candidate_images=cache['global_candidate_images'],total_unique_images=cache['total_unique_images'],encoding_seconds=cache['encoding_seconds'],api_calls=0)
        atomic_write_json(a.research/'attr_slot_017_preflight'/(a.dataset+'_native.json'),report);print(json.dumps(report),flush=True);return
    a.output.mkdir(parents=True);atomic_write_json(a.output/'protocol.json',dict(evidence=evidence,cache_sha256=file_sha(a.cache),scope='Conditional inherited legal top20 pools; NCE dev used for selection, not official FashionIQ or independent holdout.',arms=ARMS,selection='0/25/50 common retrieval-only dev NCE checkpoint per arm; all initial/final rows retained.',base_frozen=True,api_calls=0));atomic_write_json(a.output/'manifest.json',dict(records=records,offsets=offsets,image_sha256=cache['image_sha256']))
    baseline=evaluate(None,cache,offsets['dev'],baseline=True);results={};probes={n:v.detach().clone() for n,v in list(clip.named_parameters())[::100]};global_params={id(v) for v in clip.parameters()}
    for arm,flags in ARMS.items():
        torch.manual_seed(42);model=AttrSlotCIR(clip,tokenizer,use_block_mask=flags[0],use_conflict_eraser=flags[1],use_ortho_loss=flags[2]).cuda();params={n:v for n,v in model.named_parameters() if v.requires_grad}
        if any(id(v) in global_params for v in params.values()):raise ValueError('Base entered optimizer')
        optimizer=torch.optim.AdamW(params.values(),lr=.0002,weight_decay=.01);initial={n:v.detach().cpu().clone() for n,v in params.items()};best=copy.deepcopy(initial);best_value=dev_objective(model,cache,offsets['dev']);best_step=0;initial_metrics=evaluate(model,cache,offsets['dev']);exposure=torch.zeros(128,dtype=torch.long);rng=torch.Generator().manual_seed(42);history=[dict(step=0,dev_nce=best_value)];first_grad={}
        for step in range(1,51):
            if time.monotonic()-started>3300:raise TimeoutError('Pilot leaves artifact time inside1h')
            require_space(a.output);local=torch.randperm(128,generator=rng)[:4];exposure[local]+=1;ix=local.cuda();model.train();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                o=forward(model,cache,ix);has=torch.tensor([bool(records[i]['view']['drop_attrs']) for i in ix.tolist()],device='cuda');loss=model.losses(o['q_final'],cache['targets'][ix],o['f_attr_text'],cache['t_drop'][ix],has,o['attr_valid'],cache['mask'][ix])
            if not torch.isfinite(loss['total']):raise FloatingPointError('Nonfinite attr loss')
            loss['total'].backward();norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite attr gradient')
            if step==1:first_grad={n:float(v.grad.norm()) if v.grad is not None else 0. for n,v in params.items()}
            optimizer.step()
            if step in (25,50):
                value=dev_objective(model,cache,offsets['dev']);history.append(dict(step=step,dev_nce=value,train={n:float(v.detach()) for n,v in loss.items()}))
                if value<best_value:best_value=value;best_step=step;best={n:v.detach().cpu().clone() for n,v in params.items()}
        final_delta=sum(float((v.detach().cpu()-initial[n]).square().sum()) for n,v in params.items())**.5
        with torch.no_grad():
            for n,v in params.items():v.copy_(best[n].to(v.device))
        selected=evaluate(model,cache,offsets['dev']);delta=sum(float((best[n]-initial[n]).square().sum()) for n in best)**.5
        if any(not torch.equal(v,dict(clip.named_parameters())[n]) for n,v in probes.items()) or any(p.grad is not None for p in clip.parameters()):raise ValueError('Protected base changed')
        payload=dict(trainable_state=best,evidence=evidence,flags=flags,selected_step=best_step,base_checkpoint_sha256=evidence['base_sha256']);atomic_torch_save(payload,a.output/(arm+'_head.pt'))
        results[arm]=dict(initial=initial_metrics,selected=selected,selected_step=best_step,dev_nce=best_value,history=history,trainable_parameters=sum(v.numel() for v in params.values()),parameter_delta_l2=delta,final_parameter_delta_l2=final_delta,first_gradient_norms=first_grad,exposure=exposure.tolist(),updates=50,presentations=int(exposure.sum()),head_sha256=file_sha(a.output/(arm+'_head.pt')),base_in_optimizer=False,base_probe_unchanged=True)
        atomic_write_json(a.output/(arm+'_training.json'),results[arm]);del model,optimizer,params;gc.collect();torch.cuda.empty_cache()
    result=dict(status='complete',category=a.dataset,evidence=evidence,baseline_global_clip=baseline,arms=results,api_calls=0,fit_updates=200,official_val_queries_loaded=0,full_gallery_evaluated=False,conditional_candidate_pools=True,elapsed_seconds=time.monotonic()-started,max_cuda_gib=torch.cuda.max_memory_allocated()/2**30)
    atomic_write_json(a.output/'results.json',result);atomic_write_json(a.output/'state.json',dict(status='complete'));print(json.dumps(dict(status='complete',category=a.dataset,baseline=baseline,arms={n:dict(selected_step=v['selected_step'],metrics=v['selected']) for n,v in results.items()},seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
