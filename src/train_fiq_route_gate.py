"""Train a tiny route gate on train-only frozen features, then report official val."""
import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F

import newDataset
import newModel
from newTrain import load_checkpoint,load_dqu_checkpoint
from evaluate_image_keyword_routes import encode_complete_queries,encode_gallery,autocast
from generate_structured_edits import atomic_write_json,load_fashioniq_queries
from fashioniq_noise import file_sha
from experiment_storage import require_space,atomic_torch_save
from fiq_route_gate import sample_partitions,features,RouteGate,mixed_query,contrastive_loss,shuffled_valid_alpha
from fiq_small_math import blend,metrics,ranks_from_scores


class TrainOnlyViews(newDataset.FashionIQ):
    def get_test_data(self):
        return [],[]

    @staticmethod
    def _load_json(path,default=None):
        if path and Path(path).name.startswith('image_captions_') and Path(path).name.endswith('_val.json'):
            return default
        return newDataset.FashionIQ._load_json(path,default)


def encode_fit(args,selected,evidence):
    signature = hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest()[:12]
    cachefile = args.cache/f'{args.dataset}_{signature}.pt'
    if cachefile.exists():
        result = torch.load(cachefile,map_location='cpu',weights_only=True)
        if result['evidence']!=evidence:
            raise ValueError('Training feature cache mismatch')
        return result
    require_space(args.cache,1)
    folder = args.primary/args.dataset
    base = folder/'checkpoints'/f'{args.dataset}_dqu_original_seed42_best.pt'
    adapter = folder/'checkpoints'/f'{args.dataset}_structured_off_seed42_best.pt'
    config = torch.load(adapter,map_location='cpu',weights_only=True)['config']
    if config['dataset']!=args.dataset:
        raise ValueError('Wrong category adapter')
    model = newModel.DQU_CIR(clip_checkpoint=args.clip_checkpoint,freeze_clip=True,
        adapter_rank=int(config.get('adapter_rank',256)),max_structured_weight=float(config.get('max_structured_weight',.15)),
        max_residual_norm=float(config.get('max_residual_norm',1))).cuda()
    load_dqu_checkpoint(base,model,args.dataset)
    load_checkpoint(adapter,model,args.dataset)
    model.freeze_dqu_backbone()
    model.eval()
    kwargs = dict(path=str(args.root),category=args.dataset,transform=(model.preprocess_val,model.preprocess_val),
        split='original-split',structured_train_path=str(folder/f'structured_{args.dataset}_train.json'),structured_val_path=None)
    dqu = TrainOnlyViews(**kwargs,written_keyword_source='dqu')
    qwen = TrainOnlyViews(**kwargs,written_keyword_source='qwen-add')
    assert not dqu.test_queries and not dqu.test_targets and not qwen.test_queries
    index = {r['source_index']:i for i,r in enumerate(dqu.train_data)}
    dq,qq,tg = [],[],[]
    for n,row in enumerate(selected):
        item = dqu[index[row['source_index']]]
        if (item['candidate'],item['target'])!=(row['candidate'],row['target']):
            raise ValueError('Training metadata binding changed')
        dq.append(item)
        other = dict(item,visual_query=qwen.get_query_img(row['candidate'],row['target'],stage=0)[0])
        qq.append(other)
        tg.append(dict(target_img_data=item['target_img_data']))
        if n%128==0:
            print(json.dumps(dict(stage='prepare_train_features',done=n,total=len(selected))),flush=True)
    device = torch.device('cuda')
    qd = encode_complete_queries(model,dq,16,device)
    qf = encode_complete_queries(model,qq,16,device)
    targets = encode_gallery(model,tg,16,device)
    valid = torch.tensor([r['has_structured_text'] for r in dq],dtype=torch.bool)
    c = torch.tensor([r['structured_confidence'] for r in dq],dtype=torch.float32,device=device)[:,None]
    quality = torch.tensor([r['structured_quality_features'] for r in dq],dtype=torch.float32,device=device)
    with torch.inference_mode(),autocast(device):
        confidence = model.structured_confidence_calibrator(c,quality).float().cpu().flatten()
    result = dict(evidence=evidence,qd=qd,qq=qf,targets=targets,valid=valid,confidence=confidence,
        source_ids=[r['candidate'] for r in selected],target_ids=[r['target'] for r in selected])
    args.cache.mkdir(parents=True,exist_ok=True)
    atomic_torch_save(result,cachefile)
    del dq,qq,tg,model,dqu,qwen,c,quality
    gc.collect()
    torch.cuda.empty_cache()
    return result


def fit(cache,nfit,output):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    gate = RouteGate()
    x = features(cache['qd'],cache['qq'],cache['confidence'],cache['valid'])
    opt = torch.optim.AdamW(gate.parameters(),lr=5e-4,weight_decay=.01)
    rng = torch.Generator().manual_seed(42)
    initial = torch.cat([p.detach().flatten().clone() for p in gate.parameters()])
    def loss(indices):
        ids = indices.tolist()
        alpha = gate(x[indices],cache['valid'][indices])
        q = mixed_query(cache['qd'][indices],cache['qq'][indices],alpha)
        rank = contrastive_loss(q,cache['targets'][indices],[cache['source_ids'][i] for i in ids],
            [cache['target_ids'][i] for i in ids])
        return rank+.01*(alpha[cache['valid'][indices]]-.5).square().mean() if cache['valid'][indices].any() else rank
    dev = torch.arange(nfit,len(x))
    with torch.no_grad():
        initial_dev = float(loss(dev))
    if not math.isfinite(initial_dev):
        raise FloatingPointError('Nonfinite initial internal-dev loss')
    best,best_step,best_state = initial_dev,0,copy.deepcopy(gate.state_dict())
    with (output/'training.jsonl').open('x') as log:
        log.write(json.dumps(dict(step=0,internal_dev_loss=initial_dev))+'\n')
        for step in range(1,201):
            batch = torch.randperm(nfit,generator=rng)[:64]
            opt.zero_grad(set_to_none=True)
            value = loss(batch)
            if not torch.isfinite(value):
                raise FloatingPointError('Nonfinite gate training loss')
            value.backward()
            norm = torch.nn.utils.clip_grad_norm_(gate.parameters(),1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite gate gradients')
            opt.step()
            if step%50==0:
                with torch.no_grad():
                    score = float(loss(dev))
                if not torch.isfinite(torch.tensor(score)):
                    raise FloatingPointError('Nonfinite internal-dev loss')
                if score<best-1e-6:
                    best,best_step,best_state = score,step,copy.deepcopy(gate.state_dict())
                delta = float((torch.cat([p.detach().flatten() for p in gate.parameters()])-initial).norm())
                if not math.isfinite(delta):
                    raise FloatingPointError('Nonfinite gate parameters')
                row = dict(step=step,train_loss=float(value.detach()),internal_dev_loss=score,gradient_norm=float(norm),parameter_delta_l2=delta)
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    gate.load_state_dict(best_state)
    summary = dict(steps=200,selected_step=best_step,initial_internal_dev_loss=initial_dev,
        selected_internal_dev_loss=best,final_parameter_delta_l2=delta,official_validation_used_for_fit=False)
    atomic_torch_save(dict(model_state=gate.state_dict(),summary=summary),output/'gate.pt')
    atomic_write_json(output/'fit_complete.json',summary)
    return gate,summary


def evaluate_after_fit(args,gate,evidence):
    if not (args.output/'fit_complete.json').exists():
        raise ValueError('Fit must end before official validation is loaded')
    proof = json.loads((args.first_results/f'pilot_{args.dataset}.json').read_text())
    if proof['status']!='complete' or any(abs(v)>.0001 for v in proof['off_reproduction_error_pp'].values()):
        raise ValueError('First feature pilot must have passed off-checkpoint reproduction')
    candidates = []
    for file in args.val_cache.glob(args.dataset+'_*.pt'):
        cache = torch.load(file,map_location='cpu',weights_only=True)
        e = cache['evidence']
        if (e==proof['evidence'] and e['base_sha256']==evidence['base_sha256'] and e['adapter_sha256']==evidence['adapter_sha256']
                and e['val_structured_sha256']==file_sha(args.primary/args.dataset/f'structured_{args.dataset}_val.json')
                and e['model_source_sha256']==file_sha(Path(__file__).with_name('newModel.py'))):
            candidates.append(cache)
    if len(candidates)!=1:
        raise ValueError('Require exactly one verified first-batch validation feature cache')
    cache = candidates[0]
    x = features(cache['qd'],cache['qq'],cache['confidence'],cache['valid'])
    with torch.no_grad():
        alpha = gate(x,cache['valid'])
    shuffled = shuffled_valid_alpha(alpha,cache['valid'])
    routes = dict(dqu_written=cache['qd'],fixed_dual05=blend(cache['qd'],cache['qq'],.5),
        train_only_gate=mixed_query(cache['qd'],cache['qq'],alpha),
        gate_shuffled_control=mixed_query(cache['qd'],cache['qq'],shuffled))
    results = {}
    for name,query in routes.items():
        scores = query@cache['gallery'].T
        scores[torch.arange(len(query)),cache['sources']] = -torch.inf
        results[name] = metrics(ranks_from_scores(scores,cache['targets']))
    return dict(results=results,queries=len(x),gallery=len(cache['gallery']),alpha_mean=float(alpha.mean()),
        alpha_min=float(alpha.min()),alpha_max=float(alpha.max()),validation_cache_evidence=cache['evidence'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('shirt','toptee'),required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--primary',type=Path,required=True)
    parser.add_argument('--first-results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--clip-checkpoint',required=True)
    parser.add_argument('--cache',type=Path,default=Path('/dev/shm/fiq_gate_features_20261026'))
    parser.add_argument('--val-cache',type=Path,default=Path('/dev/shm/fiq_small_features_20261026'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Do not overwrite or blindly repeat a route-gate trial')
    require_space(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for frozen training feature encoding')
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    rows = load_fashioniq_queries(args.root,args.dataset,'train')
    heldout,heldout_hashes=set(),{}
    for pattern in ('split.*.val.json','split.*.test.json'):
        for file in sorted((args.root/'image_splits').glob(pattern)):
            heldout.update(json.loads(file.read_text()));heldout_hashes[file.name]=file_sha(file)
    if not heldout:
        raise ValueError('Official heldout image IDs must be available')
    selected_fit,selected_dev,counts=sample_partitions(rows,heldout)
    if len(selected_fit)<128 or len(selected_dev)<32:
        raise ValueError('Insufficient image-disjoint train-only gate partitions: '+str(counts))
    selected = selected_fit+selected_dev
    folder = args.primary/args.dataset
    main = json.loads((args.primary/'experiment_report.json').read_text())
    if main['status']!='complete':
        raise ValueError('Primary comparison incomplete')
    protocol = json.loads((args.primary/'protocol.json').read_text())
    if any(file_sha(Path(name))!=digest for name,digest in protocol['source_hashes'].items()):
        raise ValueError('Original dataset metadata changed')
    base = folder/'checkpoints'/f'{args.dataset}_dqu_original_seed42_best.pt'
    if file_sha(base)!=main['checkpoint_hashes'][args.dataset]:
        raise ValueError('Protected DQU checkpoint changed')
    manifest = dict(category=args.dataset,fit=selected_fit,internal_dev=selected_dev,partition_counts=counts,
        heldout_metadata_sha256=heldout_hashes,base_was_pretrained_on_full_official_train=True)
    evidence=dict(category=args.dataset,base_sha256=main['checkpoint_hashes'][args.dataset],
        adapter_sha256=file_sha(folder/'checkpoints'/f'{args.dataset}_structured_off_seed42_best.pt'),
        structured_train_sha256=file_sha(folder/f'structured_{args.dataset}_train.json'),
        manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
        source_sha256=file_sha(__file__),gate_math_sha256=file_sha(Path(__file__).with_name('fiq_route_gate.py')),
        model_source_sha256=file_sha(Path(__file__).with_name('newModel.py')),
        dataset_source_sha256=file_sha(Path(__file__).with_name('newDataset.py')),
        primary_protocol_sha256=file_sha(args.primary/'protocol.json'))
    args.output.mkdir(parents=True)
    atomic_write_json(args.output/'manifest.json',manifest)
    atomic_write_json(args.output/'protocol.json',dict(evidence=evidence,train_limit=1024,dev_limit=256,
        seed=42,steps=200,batch=64,lr=5e-4,temperature=.07,anchor_weight=.01,hidden=16,alpha_bounds=[.25,.75],
        selection='internal-dev loss, evaluate at steps0/50/100/150/200',additional_api_calls=0,
        note='Gate partitions have disjoint image IDs; frozen DQU was already trained on official train including these images. Official val labels are loaded only after fit_complete. Not an independent backbone holdout.'))
    atomic_write_json(args.output/'state.json',dict(status='encoding_train_only_features'))
    cache = encode_fit(args,selected,evidence)
    atomic_write_json(args.output/'state.json',dict(status='fitting_gate'))
    gate,summary = fit(cache,len(selected_fit),args.output)
    atomic_write_json(args.output/'state.json',dict(status='evaluating_official_val'))
    result = evaluate_after_fit(args,gate,evidence)
    require_space(args.output)
    atomic_write_json(args.output/'results.json',dict(status='complete',category=args.dataset,evidence=evidence,
        training=summary,fit_queries=len(selected_fit),internal_dev_queries=len(selected_dev),**result,
        interpretation='Exploratory single-seed train-only route selection; all predeclared rows reported, no official-val selection, shuffled control included, zero API cost.'))
    atomic_write_json(args.output/'state.json',dict(status='complete'))
    print(json.dumps(dict(status='complete',category=args.dataset,training=summary,results=result['results'])),flush=True)


if __name__=='__main__':
    main()
