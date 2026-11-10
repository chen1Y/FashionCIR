"""Fit a129-parameter candidate head on isolated train pools; official val only after fit."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from experiment_storage import require_space,atomic_torch_save
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json
from fiq_small_math import blend,metrics
from fiq_candidate_partition import validate_partitions
from fiq_candidate_head import (CandidateHead,candidate_features,fit_statistics,training_pool,
    shuffled_correction,head_ranks)


def one_cache(folder,category,expected):
    found=[]
    for path in folder.glob(category+'_*.pt'):
        value=torch.load(path,map_location='cpu',weights_only=True)
        if value['evidence']==expected:found.append((path,value))
    if len(found)!=1:raise ValueError('Require one exact matching feature cache')
    return found[0]


def prepare(args):
    root=args.research/'train_only_route_gate'/args.dataset
    gate_manifest=json.loads((root/'manifest.json').read_text())
    expected=json.loads((root/'protocol.json').read_text())['evidence']
    if hashlib.sha256(json.dumps(gate_manifest,sort_keys=True).encode()).hexdigest()!=expected['manifest_sha256']:
        raise ValueError('Original fit/dev manifest changed')
    fit,dev=gate_manifest['fit'],gate_manifest['internal_dev']
    if (len(fit),len(dev))!=(1024,256):raise ValueError('Keep fixed fit1024/dev256 manifests')
    source=Path(__file__).resolve().parent
    for name,key in [('newModel.py','model_source_sha256'),('newDataset.py','dataset_source_sha256')]:
        if file_sha(source/name)!=expected[key]:raise ValueError('Frozen feature encoder source changed')
    main=json.loads((args.primary/'experiment_report.json').read_text())
    if (main['status']!='complete' or main['checkpoint_hashes'][args.dataset]!=expected['base_sha256']
            or file_sha(args.primary/'protocol.json')!=expected['primary_protocol_sha256']):
        raise ValueError('Primary comparison incomplete or protocol changed')
    base=args.primary/args.dataset/'checkpoints'/f'{args.dataset}_dqu_original_seed42_best.pt'
    adapter=args.primary/args.dataset/'checkpoints'/f'{args.dataset}_structured_off_seed42_best.pt'
    if file_sha(base)!=expected['base_sha256'] or file_sha(adapter)!=expected['adapter_sha256']:
        raise ValueError('Protected base/adapter SHA changed')
    query_path,query=one_cache(args.query_cache,args.dataset,expected)
    rows=fit+dev
    if query['source_ids']!=[r['candidate'] for r in rows] or query['target_ids']!=[r['target'] for r in rows]:
        raise ValueError('Frozen query row identity changed')
    diagnostic=args.research/'train_gallery_diagnostic'/args.dataset
    gallery_manifest=json.loads((diagnostic/'manifest.json').read_text())
    completed=json.loads((args.research/'train_gallery_diagnostic/completion_summary.json').read_text())
    if file_sha(diagnostic/'manifest.json')!=completed['categories'][args.dataset]['manifest_sha256']:
        raise ValueError('Original full-gallery manifest changed')
    gallery_evidence=json.loads((diagnostic/'protocol.json').read_text())['evidence']
    gallery_path,cache=one_cache(args.gallery_cache,args.dataset,gallery_evidence)
    if (cache['ids']!=gallery_manifest['gallery_ids'] or gallery_evidence['inherited_cache_evidence']!=expected
            or file_sha(query_path)!=gallery_evidence['query_cache_sha256']):
        raise ValueError('Frozen training gallery binding changed')
    heldout=set()
    for filename,digest in gallery_evidence['metadata_sha256'].items():
        path=args.root/'image_splits'/filename
        if file_sha(path)!=digest:raise ValueError('Dataset split metadata changed')
        if '.val.' in filename or '.test.' in filename:heldout.update(json.loads(path.read_text()))
    if not heldout:raise ValueError('Require all official-heldout exclusion metadata')
    fit_bank,dev_bank=validate_partitions(fit,dev,cache['ids'],heldout)
    for key in ('qd','qq','confidence'):
        if not torch.isfinite(query[key]).all():raise FloatingPointError('Nonfinite frozen query features')
    if not torch.isfinite(cache['gallery']).all():raise FloatingPointError('Nonfinite frozen gallery')
    index={image_id:i for i,image_id in enumerate(cache['ids'])}
    sources=torch.tensor([index[r['candidate']] for r in rows]);targets=torch.tensor([index[r['target']] for r in rows])
    if not torch.equal(cache['gallery'][targets],query['targets']):
        raise ValueError('Cached gallery no longer reproduces every inherited positive feature')
    pools={}
    for name,start,end,bank in [('fit',0,1024,fit_bank),('dev',1024,1280,dev_bank)]:
        allowed=torch.zeros(len(index),dtype=torch.bool);allowed[bank]=True
        pool,scores=training_pool(query['qd'][start:end],query['qq'][start:end],cache['gallery'],
            sources[start:end],targets[start:end],allowed,rows[start:end],cache['ids'])
        if not allowed[pool].all():raise ValueError('Hard-negative image partition leakage')
        x=candidate_features(query['qd'][start:end],query['qq'][start:end],cache['gallery'][sources[start:end]],
            cache['gallery'][pool],query['confidence'][start:end])
        pools[name]=dict(indices=pool,base=scores,x=x,valid=query['valid'][start:end])
    evidence=dict(category=args.dataset,query_cache_sha256=file_sha(query_path),gallery_cache_sha256=file_sha(gallery_path),
        inherited_query_evidence=expected,inherited_gallery_evidence=gallery_evidence,
        gate_manifest_sha256=file_sha(root/'manifest.json'),source_sha256=file_sha(__file__),
        head_source_sha256=file_sha(source/'fiq_candidate_head.py'),partition_source_sha256=file_sha(source/'fiq_candidate_partition.py'))
    manifest=dict(fit=fit,internal_dev=dev,fit_gallery_ids=[cache['ids'][i] for i in fit_bank],
        dev_gallery_ids=[cache['ids'][i] for i in dev_bank],official_heldout_excluded=True,
        candidate_pool_policy='GT+31 fixed hard negatives for loss only; exact known alternative positives and reference excluded. No GT injection in retrieval evaluation.')
    return dict(query=query,gallery=cache['gallery'],ids=cache['ids'],sources=sources,targets=targets,
        pools=pools,evidence=evidence,manifest=manifest,counts=dict(fit=1024,dev=256,fit_gallery=len(fit_bank),dev_gallery=len(dev_bank)))


def fit(args,context):
    train,dev=context['pools']['fit'],context['pools']['dev']
    mean,std=fit_statistics(train['x']);head=CandidateHead(mean,std)
    optimizer=torch.optim.AdamW(head.parameters(),lr=5e-4,weight_decay=.01)
    rng=torch.Generator().manual_seed(42)
    initial=torch.cat([p.detach().flatten().clone() for p in head.parameters()])
    def loss(pool,indices=None):
        x,base,valid=(pool[k] for k in ('x','base','valid'))
        if indices is not None:x,base,valid=x[indices],base[indices],valid[indices]
        correction=head(x,valid)
        return F.cross_entropy((base+correction)/.07,torch.zeros(len(x),dtype=torch.long))+.01*correction.square().mean()
    with torch.no_grad():initial_dev=float(loss(dev))
    if not math.isfinite(initial_dev):raise FloatingPointError('Nonfinite initial internal-dev loss')
    best,best_step,best_state=initial_dev,0,copy.deepcopy(head.state_dict())
    with (args.output/'training.jsonl').open('x') as stream:
        stream.write(json.dumps(dict(step=0,internal_dev_loss=initial_dev))+'\n')
        for step in range(1,201):
            indices=torch.randperm(1024,generator=rng)[:64]
            optimizer.zero_grad(set_to_none=True);value=loss(train,indices)
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite candidate training loss')
            value.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite candidate gradients')
            optimizer.step()
            if step%50==0:
                with torch.no_grad():score=float(loss(dev))
                delta=float((torch.cat([p.detach().flatten() for p in head.parameters()])-initial).norm())
                if not math.isfinite(score) or not math.isfinite(delta):raise FloatingPointError('Nonfinite candidate parameters/dev loss')
                if score<best-1e-6:best,best_step,best_state=score,step,copy.deepcopy(head.state_dict())
                row=dict(step=step,train_loss=float(value.detach()),internal_dev_loss=score,gradient_norm=float(norm),parameter_delta_l2=delta)
                stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True)
    head.load_state_dict(best_state)
    selected_delta=float((torch.cat([p.detach().flatten() for p in head.parameters()])-initial).norm())
    summary=dict(steps=200,selected_step=best_step,initial_internal_dev_loss=initial_dev,selected_internal_dev_loss=best,
        final_parameter_delta_l2=delta,selected_parameter_delta_l2=selected_delta,trainable_parameters=sum(p.numel() for p in head.parameters()),
        official_validation_used_for_fit=False)
    atomic_torch_save(dict(model_state=head.state_dict(),summary=summary,evidence=context['evidence']),args.output/'head.pt')
    summary['checkpoint_sha256']=file_sha(args.output/'head.pt')
    atomic_write_json(args.output/'fit_complete.json',summary)
    return head,summary


def evaluate(head,cache,stable):
    qd,qq,gallery=cache['qd'],cache['qq'],cache['gallery']
    sources,targets=cache['sources'],cache['targets']
    if ((targets<0)|(targets>=len(gallery))|(sources<0)|(sources>=len(gallery))|(targets==sources)).any():
        raise ValueError('Invalid reference/target gallery bindings')
    fixed=blend(qd,qq,.5);scores=fixed@gallery.T;scores[torch.arange(len(qd)),sources]=-torch.inf
    order=torch.argsort(scores,descending=True,stable=stable);indices=order[:,:20]
    x=candidate_features(qd,qq,gallery[sources],gallery[indices],cache['confidence'])
    with torch.no_grad():correction=head(x,cache['valid'])
    if not torch.isfinite(correction).all() or float(correction.abs().max())>.03000001:
        raise FloatingPointError('Nonfinite/unbounded ranking correction')
    baseline=(order==targets[:,None]).nonzero(as_tuple=False)[:,1]+1
    if len(baseline)!=len(qd):raise ValueError('A query lost its annotated target rank')
    base_head=scores.gather(1,indices)
    ranks=dict(fixed_dual05=baseline,candidate_head=head_ranks(order,targets,base_head+correction),
        shuffled_candidate_control=head_ranks(order,targets,base_head+shuffled_correction(correction)))
    dqu_scores=qd@gallery.T;dqu_scores[torch.arange(len(qd)),sources]=-torch.inf
    dqu_order=torch.argsort(dqu_scores,descending=True,stable=stable)
    ranks['v7_off_dqu_written']=(dqu_order==targets[:,None]).nonzero(as_tuple=False)[:,1]+1
    for name in ('candidate_head','shuffled_candidate_control'):
        if not torch.equal((ranks[name]<=20),(baseline<=20)) or not torch.equal((ranks[name]<=50),(baseline<=50)):
            raise ValueError('Head-only correction changed tail recall')
    return dict(metrics={name:metrics(value) for name,value in ranks.items()},ranks={name:value.tolist() for name,value in ranks.items()},
        correction_mean=float(correction.mean()),correction_abs_max=float(correction.abs().max()),queries=len(qd),gallery=len(gallery))


def load_val_after_fit(args,evidence):
    if not (args.output/'fit_complete.json').exists():raise ValueError('Do not load official validation features before fit_complete')
    fit_summary=json.loads((args.output/'fit_complete.json').read_text())
    if file_sha(args.output/'head.pt')!=fit_summary['checkpoint_sha256']:raise ValueError('Selected head changed after fit')
    proof=json.loads((args.research/f'pilot_{args.dataset}.json').read_text())
    if proof['status']!='complete' or any(abs(v)>.0001 for v in proof['off_reproduction_error_pp'].values()):
        raise ValueError('Original frozen pilot failed off reproduction')
    expected=proof['evidence'];source=Path(__file__).resolve().parent
    if (expected['base_sha256']!=evidence['inherited_query_evidence']['base_sha256']
            or expected['adapter_sha256']!=evidence['inherited_query_evidence']['adapter_sha256']
            or expected['model_source_sha256']!=file_sha(source/'newModel.py')
            or expected['val_structured_sha256']!=file_sha(args.primary/args.dataset/f'structured_{args.dataset}_val.json')):
        raise ValueError('Official validation cache evidence changed')
    path,cache=one_cache(args.val_cache,args.dataset,expected)
    if (len(cache['qd']),len(cache['gallery']))!={'shirt':(2038,6346),'toptee':(1961,5373)}[args.dataset]:
        raise ValueError('Incomplete official validation queries/gallery')
    for key in ('qd','qq','gallery','confidence'):
        if not torch.isfinite(cache[key]).all():raise FloatingPointError('Nonfinite official validation features')
    return cache,proof,dict(cache_sha256=file_sha(path),evidence=expected)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('shirt','toptee'),required=True)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--primary',type=Path,required=True)
    parser.add_argument('--research',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--query-cache',type=Path,default=Path('/dev/shm/fiq_gate_features_20261026'))
    parser.add_argument('--gallery-cache',type=Path,default=Path('/dev/shm/fiq_train_gallery_20261026'))
    parser.add_argument('--val-cache',type=Path,default=Path('/dev/shm/fiq_small_features_20261026'))
    parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Inspect existing candidate head; never blindly repeat or overwrite')
    require_space(args.output);torch.set_num_threads(4);torch.manual_seed(42)
    context=prepare(args)
    if args.preflight:
        print(json.dumps(dict(status='preflight_passed',counts=context['counts'],pool_shapes={n:list(p['x'].shape) for n,p in context['pools'].items()},
            official_val_features_loaded=False,api_calls=0)),flush=True);return
    args.output.mkdir(parents=True)
    atomic_write_json(args.output/'protocol.json',dict(evidence=context['evidence'],seed=42,steps=200,lr=5e-4,batch=64,
        hidden=16,temperature=.07,anchor=.01,delta_bound=.03,statistics='Fit-only mean/std, std>=1e-3',
        selection='Internal-dev NCE loss at0/50/100/150/200; validation loaded only after fit_complete',
        negatives='GT+31 fixed hard negatives; both positive and negative images partition-isolated; known alternative positives excluded',
        official_eval='Complete original validation/gallery, reference excluded, top20 correction with matched per-query shuffle',api_calls=0,
        limitation='Backbone saw full official train; dev query/target supervision is isolated but full retrieval galleries share candidates. Unannotated reasonable images may be false negatives.'))
    atomic_write_json(args.output/'manifest.json',context['manifest'])
    atomic_torch_save({n:p['indices'] for n,p in context['pools'].items()},args.output/'candidate_pools.pt')
    atomic_write_json(args.output/'state.json',dict(status='fitting_train_only_candidate_head'))
    head,training=fit(args,context)
    atomic_write_json(args.output/'state.json',dict(status='evaluating_after_fit'))
    internal=dict(qd=context['query']['qd'][1024:],qq=context['query']['qq'][1024:],gallery=context['gallery'],
        sources=context['sources'][1024:],targets=context['targets'][1024:],confidence=context['query']['confidence'][1024:],valid=context['query']['valid'][1024:])
    internal_result=evaluate(head,internal,True)
    cache,proof,val_evidence=load_val_after_fit(args,context['evidence']);result=evaluate(head,cache,False)
    for row,old in [('fixed_dual05','fixed_dual_alpha05'),('v7_off_dqu_written','v7_off_dqu_written')]:
        if any(abs(result['metrics'][row][k]-proof['results'][old][k])>.0001 for k in ('r1','r10','r50')):
            raise ValueError('Frozen official baseline does not reproduce; do not report head gains')
    require_space(args.output)
    atomic_write_json(args.output/'results.json',dict(status='complete',category=args.dataset,training=training,counts=context['counts'],
        evidence=context['evidence'],validation_evidence=val_evidence,internal_dev_retrieval=internal_result,official_validation=result,
        fixed_baselines_reproduced=True,api_calls=0,scope='Exploratory single-seed candidate-level head; all rows retained; no official-val fitting or significance claim.'))
    atomic_write_json(args.output/'state.json',dict(status='complete'))
    print(json.dumps(dict(status='complete',category=args.dataset,training=training,official_metrics=result['metrics'])),flush=True)


if __name__=='__main__':main()
