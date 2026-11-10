"""018 two preregistered text-prealignment arms on actual reference/target pairs."""
from __future__ import annotations
import argparse,copy,gc,json,time,hashlib,os
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
import train_attr_slot_pilot as old
from attr_slot_data import SLOTS
from attr_token_pair_alignment import PairAlignedText,paired_losses,pair_query,packed_audit
from attr_slot_rank_math import candidate_ranks
from experiment_storage import require_space,atomic_torch_save
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json

TASK_SHA='e0c34eeefa8e1c1c8118ad40f98822294465eb8edccce9b201d542c1b2691d00'
SOURCE_NAMES=('attr_token_pair_alignment.py','train_attr_pair_alignment.py','attr_slot_fusion.py','attr_slot_data.py','train_attr_slot_pilot.py','attr_slot_rank_math.py')

def frozen_digest(clip):
    h=hashlib.sha256()
    for n,p in clip.named_parameters():
        if not p.requires_grad:
            h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def guard(a,started):
    require_space(a.research)
    if time.monotonic()-started>3300:raise TimeoutError('018 one category exceeds3300s')
    if (a.output.parent/'cancel.request').exists():raise KeyboardInterrupt('User stopped018')

def encode(a,clip,transform,tokenizer,records,hashes,evidence):
    if a.cache.exists():raise FileExistsError('No cache overwrite or blind reencoding')
    patch_ids=set(r['reference'] for r in records)|set(r['target'] for r in records)
    ids=list(dict.fromkeys(i for r in records for i in [r['reference']]+r['pool_ids']))
    vs={};patches={};checked={};start=time.monotonic();clip.visual.output_tokens=True
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        # Every unique image is read and encoded once in this new finite subset.
        for j in range(0,len(ids),8):
            batch_ids=ids[j:j+8];ims=[]
            for image_id in batch_ids:
                p=a.root/'resized_image'/(image_id+'.jpg')
                if file_sha(p)!=hashes[image_id]:raise ValueError('JPEG source changed')
                checked[image_id]=hashes[image_id]
                with Image.open(p) as im:ims.append(transform(im.convert('RGB')))
            global_,tokens=clip.visual(torch.stack(ims).cuda())
            if clip.visual.final_ln_after_pool:tokens=clip.visual.ln_post(tokens)
            projected=tokens@clip.visual.proj
            for k,image_id in enumerate(batch_ids):
                vs[image_id]=F.normalize(global_[k].float(),dim=-1).cpu()
                if image_id in patch_ids:patches[image_id]=F.normalize(projected[k].float(),dim=-1).half().cpu()
        clip.visual.output_tokens=False
        def text(values):
            out=[]
            for j in range(0,len(values),16):out.append(F.normalize(clip.encode_text(tokenizer(values[j:j+16]).cuda()).float(),dim=-1).cpu())
            return torch.cat(out)
        mod=text([r['view']['mod_text'] for r in records])
        attr=text([r['view']['target_attrs'][s] or 'unspecified' for r in records for s in SLOTS]).reshape(len(records),4,-1)
    teachers=[]
    for i,r in enumerate(records):
        p=patches[r['target']].float();weights=torch.softmax(attr[i]@p.T/.07,dim=-1)
        teachers.append(F.normalize(weights@p,dim=-1))
    pool=torch.zeros(len(records),20,1024);mask=torch.zeros(len(records),20,dtype=torch.bool);order=torch.full((len(records),20),99,dtype=torch.long)
    for i,r in enumerate(records):
        n=len(r['pool_ids']);pool[i,:n]=torch.stack([vs[x] for x in r['pool_ids']]);mask[i,:n]=True;order[i,:n]=torch.tensor(r['original_order'])
    cache=dict(records=records,evidence=evidence,image_sha256=checked,targets=pool,mask=mask,original_order=order,
               v_ref=torch.stack([vs[r['reference']] for r in records]),mod=mod,slot_teacher=torch.stack(teachers),
               unique_images=len(ids),encoding_seconds=time.monotonic()-start,teacher_scope='Frozen attribute-text weighted TARGET-image patches; soft cue, not human attribute GT.')
    if any(not torch.isfinite(cache[k]).all() for k in ('targets','v_ref','mod','slot_teacher')):raise FloatingPointError('Nonfinite cache')
    atomic_torch_save(cache,a.cache);return cache

def output(model,cache,ix):return model([cache['records'][i]['view']['target_attrs'] for i in ix.tolist()])

def evaluate(model,cache,indices,baseline=False):
    ranks=[];sums={k:0. for k in ('retrieval','global_alignment','slot_alignment')}
    with torch.no_grad():
        for j in range(0,len(indices),4):
            ix=torch.tensor(indices[j:j+4],device='cuda');n=len(ix)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if baseline:q=F.normalize(cache['v_ref'][ix]+cache['mod'][ix],dim=-1)
                else:
                    out=output(model,cache,ix);q=pair_query(out,cache['v_ref'][ix],cache['mod'][ix])
                    losses=paired_losses(out,cache['v_ref'][ix],cache['mod'][ix],cache['targets'][ix],cache['mask'][ix],cache['slot_teacher'][ix])
                    for k in sums:sums[k]+=float(losses[k])*n
            score=torch.einsum('bd,bmd->bm',q.float(),cache['targets'][ix]);ranks+=candidate_ranks(score,cache['mask'][ix],cache['original_order'][ix]).tolist()
            if baseline:sums['retrieval']+=float(F.cross_entropy(score.masked_fill(~cache['mask'][ix],-torch.inf)/.07,torch.zeros(n,dtype=torch.long,device='cuda')))*n
    return dict(queries=len(indices),ranks=ranks,candidate_pool_r1=100*sum(r<=1 for r in ranks)/len(ranks),candidate_pool_r5=100*sum(r<=5 for r in ranks)/len(ranks),candidate_pool_r10=100*sum(r<=10 for r in ranks)/len(ranks),**{k:v/len(indices) for k,v in sums.items()})

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('shirt','toptee'),required=True)
    for n in ('root','primary','research','output','cache'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');a=p.parse_args();a.task_sha256=TASK_SHA
    started=time.monotonic();torch.set_num_threads(4);torch.manual_seed(42);guard(a,started)
    if a.output.exists():raise FileExistsError('Preserve all completed/partial018 output')
    rows,offsets,hashes,base,evidence=old.metadata(a)
    evidence.update(source018={n:file_sha(Path(__file__).with_name(n)) for n in SOURCE_NAMES})
    pre=a.research/'attr_pair_018_preflight';pre.mkdir(exist_ok=True)
    # Full source binding precedes model loading and all fitting.
    clip,transform,tokenizer=old.load_clip(base)
    if a.cache.exists():
        cache=torch.load(a.cache,map_location='cpu',weights_only=True)
        if cache['records']!=rows or cache['evidence']!=evidence:raise ValueError('018 cache binding changed')
    elif a.preflight:cache=encode(a,clip,transform,tokenizer,rows,hashes,evidence)
    else:raise ValueError('No sealed real cache preflight; never silently encode during training')
    cache_sha=file_sha(a.cache);cache={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in cache.items()}
    if a.preflight:
        m=PairAlignedText(clip,tokenizer,2).cuda();examples=packed_audit(m,[r['view']['target_attrs'] for r in rows[:8]])
        for e,r in zip(examples,rows[:8]):e.update(source_index=r['source_index'],reference=r['reference'],target=r['target'],modification_text=r['view']['mod_text'],view_trace=r['view']['trace'])
        atomic_write_json(pre/(a.dataset+'_packed_examples.json'),examples)
        baseline=evaluate(None,cache,offsets['dev'],True);initial=evaluate(m,cache,offsets['dev']);before=m.embedding.weight[:m.embedding.base_size].detach().clone()
        ix=torch.tensor(offsets['fit'][:4],device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=output(m,cache,ix);loss=paired_losses(out,cache['v_ref'][ix],cache['mod'][ix],cache['targets'][ix],cache['mask'][ix],cache['slot_teacher'][ix])
        loss['total'].backward();g=m.embedding.weight.grad
        grads={n:float(p.grad.norm()) for n,p in clip.named_parameters() if p.requires_grad and p.grad is not None}
        if g[:m.embedding.base_size].count_nonzero() or not (g[m.embedding.base_size:].norm(dim=1)>0).all() or not grads or not all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):raise ValueError('Native hook/last blocks gradient failed')
        if any(p.grad is not None for p in clip.visual.parameters()):raise ValueError('ViT received gradients')
        opt=torch.optim.AdamW(m.parameter_groups());opt.step()
        if not torch.equal(before,m.embedding.weight[:m.embedding.base_size]):raise ValueError('AdamW changed old vocabulary')
        report=dict(status='preflight_passed',category=a.dataset,evidence=evidence,cache_sha256=cache_sha,baseline=baseline,initial=initial,
            new_row_gradient_norms=g[m.embedding.base_size:].norm(dim=1).tolist(),old_row_gradient_nonzeros=0,old_rows_exact_after_optimizer_step=True,
            last_block_gradient_norms=grads,vit_gradients=False,text_layers=len(clip.transformer.resblocks),opened_last_blocks=2,
            embedding_shape=list(m.embedding.weight.shape),api_calls=0,preflight_smoke_updates=1,preflight_weights_discarded=True,
            unique_images=cache['unique_images'],encoding_seconds=cache['encoding_seconds'])
        atomic_write_json(pre/(a.dataset+'_native.json'),report);print(json.dumps(report),flush=True);return
    report=json.loads((pre/(a.dataset+'_native.json')).read_text())
    if report['cache_sha256']!=cache_sha or report['evidence']!=evidence:raise ValueError('Preflight no longer matches')
    # Reserve output capacity for ONE final last-two-block delta (approximately100MiB).
    require_space(a.research,min_free_gib=8.12)
    a.output.mkdir(parents=True);atomic_write_json(a.output/'protocol.json',dict(evidence=evidence,cache_sha256=cache_sha,api_calls=0,
        fit=128,dev=32,steps=50,batch=4,seed=42,arms=['tokens_only','tokens_last2'],token_lr=.0002,block_lr=.00001,embedding_weight_decay=0.,block_weight_decay=.01,
        selection='retrieval-only internal dev NCE at0/25/50; includes initial',loss='retrieval+.1*global_target_alignment+.1*soft_target_patch_slot_alignment',
        scope='Same inherited011 legal conditional pools, no official val/full gallery; post-hoc exploratory; no expert attribute GT. No eraser/fusion training.'))
    atomic_write_json(a.output/'manifest.json',dict(records=rows,offsets=offsets,image_sha256=cache['image_sha256']))
    baseline=evaluate(None,cache,offsets['dev'],True);results={};initial_reference=None
    for arm,blocks in [('tokens_only',0),('tokens_last2',2)]:
        torch.manual_seed(42);m=PairAlignedText(clip,tokenizer,blocks).cuda();m.eval()
        frozen_before=frozen_digest(clip);initial=m.delta_state();best=copy.deepcopy(initial)
        init_metrics=evaluate(m,cache,offsets['dev']);best_value=init_metrics['retrieval'];selected_step=0;history=[dict(step=0,metrics=init_metrics)]
        if initial_reference is None:initial_reference=init_metrics
        elif init_metrics!=initial_reference:raise ValueError('Two arms do not share initial outputs')
        opt=torch.optim.AdamW(m.parameter_groups());parameters=[p for group in opt.param_groups for p in group['params']]
        rng=torch.Generator().manual_seed(42);exposure=torch.zeros(128,dtype=torch.long);oldrows=m.embedding.base_size;first_grad={}
        for step in range(1,51):
            guard(a,started);local=torch.randperm(128,generator=rng)[:4];exposure[local]+=1;ix=local.cuda();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=output(m,cache,ix);loss=paired_losses(out,cache['v_ref'][ix],cache['mod'][ix],cache['targets'][ix],cache['mask'][ix],cache['slot_teacher'][ix])
            if not torch.isfinite(loss['total']):raise FloatingPointError('Nonfinite paired loss')
            loss['total'].backward()
            if m.embedding.weight.grad[:oldrows].count_nonzero():raise ValueError('Hook leaked to old vocabulary')
            if any(p.grad is not None for p in clip.parameters() if not p.requires_grad):raise ValueError('Frozen parameter received gradients')
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            if step==1:first_grad=dict(new_tokens=m.embedding.weight.grad[-4:].norm(dim=1).tolist(),blocks={n:float(p.grad.norm()) for n,p in clip.named_parameters() if p.requires_grad and p.grad is not None})
            opt.step()
            if not torch.equal(m.embedding.weight[:oldrows],clip.token_embedding.weight):raise ValueError('Old embedding drift after optimizer step')
            if step in (25,50):
                metrics=evaluate(m,cache,offsets['dev']);history.append(dict(step=step,metrics=metrics,train={k:float(v.detach()) for k,v in loss.items()}))
                atomic_write_json(a.output/(arm+'_progress.json'),dict(step=step,history=history))
                if metrics['retrieval']<best_value:best_value=metrics['retrieval'];selected_step=step;best=m.delta_state()
        if frozen_digest(clip)!=frozen_before:raise ValueError('Frozen backbone checksum changed')
        final=m.delta_state();m.load_delta(best);selected=evaluate(m,cache,offsets['dev'])
        selected_delta=dict(tokens=float((best['new_token_rows']-initial['new_token_rows']).norm()),blocks=sum(float((best['blocks'][n]-initial['blocks'][n]).square().sum()) for n in best['blocks'])**.5)
        atomic_torch_save(dict(state=best,evidence=evidence,arm=arm,selected_step=selected_step),a.output/(arm+'_delta.pt'))
        results[arm]=dict(initial=init_metrics,selected=selected,selected_step=selected_step,history=history,updates=50,presentations=int(exposure.sum()),exposure=exposure.tolist(),
            selected_delta_l2=selected_delta,first_grad=first_grad,frozen_sha256_before=frozen_before,frozen_sha256_after=frozen_before,
            old_vocabulary_exact=True,vit_frozen=True,allowed_embedding_rows=4,effective_embedding_parameters=4096,
            trainable_block_parameters=sum(p.numel() for p in clip.parameters() if p.requires_grad),delta_sha256=file_sha(a.output/(arm+'_delta.pt')))
        atomic_write_json(a.output/(arm+'_training.json'),results[arm]);del opt,parameters,m,initial,best,final;gc.collect();torch.cuda.empty_cache()
    result=dict(status='complete',category=a.dataset,baseline=baseline,arms=results,evidence=evidence,cache_sha256=cache_sha,api_calls=0,official_val_queries_loaded=0,full_gallery_evaluated=False,elapsed_seconds=time.monotonic()-started,max_cuda_gib=torch.cuda.max_memory_allocated()/2**30)
    atomic_write_json(a.output/'results.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('arms','evidence')}),flush=True)

if __name__=='__main__':main()
