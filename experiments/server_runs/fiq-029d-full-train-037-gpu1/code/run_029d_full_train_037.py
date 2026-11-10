"""Tune 029D with an internal dev split, then refit on every eligible train row."""

import argparse
import gc
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import torch

from attr_transfer_data import SLOTS, slots_for, text_path
from expanded_attr_model import legal_masks
from experiment_storage import atomic_torch_save, require_space
from fiq_fusion_029 import encode_parts, fuse_parts, objective
from fiq_native_caption_math_031 import metrics, transitions
from run_029d_freeze_patch import eval_global, ranks_for
from run_029d_full_clip_034 import CATEGORIES, official_bank
from run_029d_openclip_036 import OPENCLIP_SHA, from_public_clip, gpu_snapshot
from run_attr_transfer_029d import ARM, encode_bank, training_context
from run_fiq_native_caption_031 import BASE, identity, read, sha, write
from train_attr_pair_alignment import frozen_digest
from train_fiq_fusion_029 import evaluate


COUNTS = {'dress': (5985,4078,1561), 'shirt': (5988,4114,1198), 'toptee': (6027,3992,1299)}
SEED = 42
BATCH = 16
PATIENCE = 3
RATES = {'lr1x': 1., 'lr2x': 2.}
RUNS = BASE/'runs'


def full_text_path(category):
    if category == 'dress':
        return text_path(category,'train')
    return BASE/'autodl-tmp/results/fiq_shirt_toptee_20261022'/category/f'structured_{category}_train.json'


def prepare(root):
    source = RUNS/'fiq-full-024'
    inventory = read(source/'full_train_inventory.json')['categories']
    prepared = root/'metadata'
    prepared.mkdir()
    summary = {}
    for cat in CATEGORIES:
        source_meta = source/'metadata'/f'{cat}_train.json'
        if sha(source_meta) != inventory[cat]['metadata_sha256']:
            raise ValueError(f'{cat} sealed metadata changed')
        meta = read(source_meta)
        n,fit,dev = COUNTS[cat]
        if (meta['original_train_rows'],len(meta['offsets']['fit']),len(meta['offsets']['dev'])) != (n,fit,dev):
            raise ValueError(f'{cat} train/fit/dev counts changed')
        if any(meta['byte_intersections'].values()) or len(meta['records']) != fit+dev:
            raise ValueError(f'{cat} isolated partition invalid')
        archived = full_text_path(cat)
        samples = read(archived)['samples']
        indexed = {r['source_index']:r for r in samples}
        if len(samples) != n or set(indexed) != set(range(n)):
            raise ValueError(f'{cat} full structured-text coverage changed')
        inputs = {str(archived):sha(archived),str(source_meta):sha(source_meta)}
        direct = {}
        if cat != 'dress':
            path = RUNS/'fiq-fusion-029/texts'/f'{cat}_train.json'
            evidence = read(RUNS/'fiq-fusion-029'/f'assets_{cat}_train.json')
            if sha(path) != evidence['text_sha256']:
                raise ValueError(f'{cat} 036 attributes changed')
            direct = {r['source_index']:r for r in read(path)['samples']}
            inputs[str(path)] = evidence['text_sha256']
        origins = Counter()
        for row in meta['records']:
            key = row['source_index']
            sample = indexed[key]
            if (sample['category'] != cat or sample['split'] != 'train'
                    or any(sample[k] != row[k] for k in ('candidate','target','modification_text'))):
                raise ValueError(f'{cat} archived edit mismatch at {key}')
            attrs = slots_for(sample)
            if any(attrs.values()):
                binding = sample['generation_input']
                if (binding.get('target_was_provided_to_model') is not False
                        or binding['modification_text'] != row['modification_text']
                        or Path(binding['reference_image']).stem != row['candidate']):
                    raise ValueError(f'{cat} archived edit provenance mismatch at {key}')
            origin = 'structured_archive' if any(attrs.values()) else 'raw_fallback'
            if key in direct:
                d = direct[key]
                if (d['reference'] != row['candidate'] or d['instruction'] != row['modification_text']
                        or d['reference_sha256'] != meta['image_sha256'][row['candidate']]):
                    raise ValueError(f'{cat} direct-slot provenance mismatch at {key}')
                if any(d['target_attrs'].values()):
                    attrs = {s:d['target_attrs'][s] for s in SLOTS}
                    origin = 'direct_036'
            row['target_attrs'] = attrs
            origins[origin] += 1
        keep = ('records','offsets','banks','canonical','image_sha256','known_positive_sha')
        payload = {key:meta[key] for key in keep}
        output = prepared/f'{cat}.json'
        write(output,payload)
        summary[cat] = dict(original=n,fit=fit,dev=dev,refit=fit+dev,excluded=n-fit-dev,
                            gallery=len(set(meta['banks']['fit']+meta['banks']['dev'])),
                            attribute_origins=dict(origins),input_sha256=inputs,metadata_sha256=sha(output))
        print(cat,summary[cat],flush=True)
    write(root/'inventory.json',summary)


def recall_score(values):
    # All checkpoints in a category use the same dev queries and denominator.
    return sum(int(r <= 10)+int(r <= 50) for r in values['ranks'])


def full_fit_bank(cache, meta):
    ids = list(dict.fromkeys(meta['banks']['fit']+meta['banks']['dev']))
    canonical = list(dict.fromkeys(meta['canonical']['fit']+meta['canonical']['dev']))
    full = dict(meta,banks={'fit':ids},canonical={'fit':canonical},offsets={'fit':list(range(len(meta['records'])))})
    mask, labels = legal_masks(full,'fit')
    return dict(indices=full['offsets']['fit'],gallery=cache['gallery'][[cache['positions'][i] for i in ids]],
                labels=labels.cuda(),mask=mask.cuda())


def fit_phase(directory, model, cache, bank, scale, progress, *, dev_context=None, epochs=None):
    directory.mkdir()
    torch.manual_seed(SEED)
    groups = model.parameter_groups()
    for group in groups:
        group['lr'] *= scale
    params = [p for group in groups for p in group['params']]
    optimizer = torch.optim.AdamW(groups)
    rng = torch.Generator().manual_seed(SEED)
    frozen = frozen_digest(model.clip)
    n = len(bank['indices'])
    exposure = torch.zeros(n,dtype=torch.long)
    history = []
    best_epoch, best_score, stale, updates, epoch = 0,-1,0,0,0

    def record():
        nonlocal best_epoch,best_score,stale
        progress('dev',epoch,0,stale=stale,updates=updates)
        values = evaluate(model,dev_context,'dev',ARM)
        score = recall_score(values)
        if score > best_score:
            best_epoch,best_score,stale = epoch,score,0
        else:
            stale += 1
        history.append(dict(epoch=epoch,updates=updates,score=score,stale=stale,metrics=values))
        write(directory/'history.json',dict(best_epoch=best_epoch,best_score=best_score,history=history))

    if dev_context is not None:
        record()
    epoch_limit = math.inf if dev_context is not None else epochs
    while epoch < epoch_limit and stale < PATIENCE:
        epoch += 1
        model.train()
        sums = Counter()
        order = torch.randperm(n,generator=rng)
        for local in order.split(BATCH):
            positions = [bank['indices'][i] for i in local.tolist()]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                parts = encode_parts(model,[cache['rows'][i] for i in positions],ARM)
                out = fuse_parts(model,parts,cache['reference'][positions],cache['written'][positions],ARM)
                loss = objective(out,bank['gallery'],bank['labels'][local],bank['mask'][local],cache['teacher'][positions],ARM)
            if not torch.isfinite(loss['total']):
                raise FloatingPointError('Nonfinite 037 objective')
            loss['total'].backward()
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step()
            if not torch.equal(model.embedding.weight[:49408],model.clip.token_embedding.weight):
                raise ValueError('Original vocabulary changed')
            exposure[local] += 1
            updates += 1
            for name,value in loss.items():
                sums[name] += float(value.detach())*len(local)
            if updates % 16 == 0 or updates % math.ceil(n/BATCH) == 0:
                progress('fit',updates,0,epoch=epoch,stale=stale)
        with (directory/'epoch_losses.jsonl').open('a',encoding='utf8') as log:
            log.write(json.dumps(dict(epoch=epoch,updates=updates,**{k:v/n for k,v in sums.items()}))+'\n')
        if dev_context is not None:
            record()
    if exposure.tolist()!=[epoch]*n or updates!=epoch*math.ceil(n/BATCH) or frozen_digest(model.clip)!=frozen:
        raise ValueError('037 exposure or frozen parameters changed')
    report = dict(status='complete',queries=n,updates=updates,epochs=epoch,best_epoch=best_epoch,
                  best_score=best_score,stale=stale,history=history,lr_multiplier=scale)
    if dev_context is None:
        path = directory/'refit_delta.pt'
        atomic_torch_save(dict(state=model.delta_state(),epoch=epoch),path)
        report.update(checkpoint=path.name,checkpoint_sha256=sha(path))
    write(directory/'fit_results.json',report)
    return report


def run(root, category, gpu):
    directory = root/category
    directory.mkdir()
    path = directory/'state.json'
    plan = read(root/'plan.json')
    gpu_index = plan['gpu_index']
    state = dict(status='running',category=category,identity=identity(),gpu=gpu,api_calls=0)
    permitted = {}

    def progress(stage,done=0,total=0,**fields):
        if (root/'cancel.request').exists():
            raise InterruptedError('037 cancelled')
        card, other = gpu_snapshot(gpu,gpu_index)
        if card['free_mib'] < 4096 or any(pid not in permitted or used > permitted[pid]+512 for pid,used in other.items()):
            state['gpu_at_stop'] = [card,other]
            raise RuntimeError('037 GPU shared capacity changed')
        state.update(stage=stage,done=done,total=total,**fields)
        write(path,state)

    try:
        require_space(root,20)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu or plan['gpu'] != gpu:
            raise RuntimeError('GPU UUID was not bound')
        for name,digest in plan['source_sha256'].items():
            if sha(root/'code'/name) != digest:
                raise ValueError(f'037 sealed source changed: {name}')
        first = gpu_snapshot(gpu,gpu_index)
        time.sleep(5)
        second = gpu_snapshot(gpu,gpu_index)
        state['gpu_startup'] = [first,second]
        if (any(g['free_mib']<14336 or g['utilization']>75 for g,_ in (first,second))
                or any(pid not in first[1] or used>first[1][pid]+512 for pid,used in second[1].items())):
            raise RuntimeError('037 GPU lacks stable shared headroom')
        permitted = {pid:max(used,first[1][pid]) for pid,used in second[1].items()}
        progress('public_clip_load')
        torch.manual_seed(SEED)
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.40)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        model,transform = from_public_clip(root/'weights/open_clip_pytorch_model.bin')
        if permitted:
            def pace(_module,_inputs,_output):
                torch.cuda.synchronize()
                time.sleep(.05)
            model.clip.visual.register_forward_hook(pace)
            model.clip.transformer.register_forward_hook(pace)
        source = root/'metadata'/f'{category}.json'
        if sha(source) != plan['metadata_sha256'][category]:
            raise ValueError('037 prepared data changed')
        meta = read(source)
        ids = list(dict.fromkeys(meta['banks']['fit']+meta['banks']['dev']))
        cache = encode_bank(model.clip,transform,model.tokenizer,category,'train',meta['records'],ids,meta['image_sha256'],progress)
        context = training_context(cache,meta)
        initial = model.delta_state()
        tuning = {}
        for name,scale in RATES.items():
            model.load_delta(initial)
            callback = lambda s,d,t,**f:progress(s,d,t,arm=name,**f)
            tuning[name] = fit_phase(directory/name,model,cache,context['banks']['fit'],scale,callback,dev_context=context)
        chosen = max(tuning,key=lambda name:tuning[name]['best_score'])
        best = tuning[chosen]
        write(directory/'selection.json',dict(profile=chosen,epoch=best['best_epoch'],score=best['best_score']))
        del context
        model.load_delta(initial)
        progress('full_refit_prepare',arm=chosen)
        full = full_fit_bank(cache,meta)
        refit = fit_phase(directory/'refit',model,cache,full,RATES[chosen],
                          lambda s,d,t,**f:progress(s,d,t,arm='refit',**f),epochs=best['best_epoch'])
        del cache,full,meta
        gc.collect()
        torch.cuda.empty_cache()
        model.eval()
        val,hashes,native = official_bank(category)
        rows = [dict(r,native_text=t) for r,t in zip(val['rows'],native)]
        bank = encode_bank(model.clip,transform,model.tokenizer,category,'val',rows,val['ids'],hashes,progress)
        scores,_ = eval_global(model,bank,False,lambda d,t:progress('official_ranks',d,t))
        output = {}
        for view,value in scores.items():
            ranks = ranks_for(value,bank)
            output[view] = dict(metrics(ranks),ranks=ranks)
        old = RUNS/'fiq-029d-openclip-036-gpu1'/category/'results.json'
        if sha(old) != plan['control_sha256'][category]:
            raise ValueError('036 control results changed')
        control = read(old)['rows']
        comparisons = {view:{str(k):transitions(m['ranks'],control[f'public_clip_{view}']['ranks'],k)
                     for k in (1,10,50)} for view,m in output.items()}
        write(directory/'results.json',dict(category=category,queries=len(rows),gallery=len(val['ids']),
              rows=output,comparisons=comparisons,selected_profile=chosen,refit=refit,tuning=tuning,
              openclip_sha256=OPENCLIP_SHA,control_sha256=plan['control_sha256'][category],api_calls=0))
        state.update(status='complete',stage='complete')
        write(path,state)
    except Exception as error:
        state.update(status='failed',error=type(error).__name__,detail=str(error))
        write(path,state)
        raise


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--category',choices=CATEGORIES)
    p.add_argument('--gpu')
    a = p.parse_args()
    if a.prepare:
        prepare(a.root)
    else:
        if not a.category or not a.gpu:
            p.error('--category and --gpu are required when fitting')
        run(a.root,a.category,a.gpu)
