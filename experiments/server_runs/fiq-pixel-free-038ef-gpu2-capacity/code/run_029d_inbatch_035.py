"""Compare 029D negative pools at batch sizes 8 and 32 on FashionIQ."""

import argparse
import gc
import math
import os
import time
from pathlib import Path

import torch

from experiment_storage import require_space
from fiq_fusion_029 import encode_parts, fuse_parts, objective
from fiq_native_caption_math_031 import metrics, transitions
from run_029d_freeze_patch import eval_global, ranks_for
from run_029d_full_clip_034 import CATEGORIES, load_sources, make_model, official_bank
from run_029d_spelling_pair import cached_bank
from run_attr_transfer_029d import ARM, encode_bank, fit, training_context
from run_fiq_native_caption_031 import identity, read, sha, snapshot, write


BATCHES = (8, 32)
ARMS = tuple((pool, batch) for batch in BATCHES for pool in ('full', 'inbatch'))


def batch_columns(labels, mask, canonical):
    """Keep each true target and expose byte-canonical batch targets as negatives."""
    representatives = canonical[labels]
    indices = torch.cat((labels, representatives[representatives >= 0])).unique(sorted=True)
    return indices, torch.searchsorted(indices, labels), mask[:, indices]


def canonical_targets(meta):
    hashes = meta['image_sha256']
    ids = meta['banks']['fit']
    legal = set(meta['canonical']['fit'])
    canonical = {hashes[key]:i for i,key in enumerate(ids) if key in legal}
    if len(canonical) != len(legal):
        raise ValueError('Expected a unique representative for each JPEG byte sequence')
    return torch.tensor([canonical.get(hashes[key], -1) for key in ids], device='cuda')


def masked_batch_objective(canonical):
    def loss(out, gallery, labels, mask, teacher, arm):
        ids, local_labels, local_mask = batch_columns(labels, mask, canonical)
        return objective(out, gallery[ids], local_labels, local_mask, teacher, arm)
    return loss


def negative_stats(bank, canonical, pool, batch):
    rng = torch.Generator().manual_seed(42)
    counts = []
    for _ in range(3):
        order = torch.randperm(len(bank['labels']), generator=rng)
        for rows in order.split(batch):
            mask = bank['mask'][rows]
            if pool == 'inbatch':
                _, _, mask = batch_columns(bank['labels'][rows], mask, canonical)
            counts.append((mask.sum(dim=1)-1).cpu())
    x = torch.cat(counts)
    return dict(mean=float(x.float().mean()), min=int(x.min()), max=int(x.max()),
                no_negative=int((x == 0).sum()), presentations=len(x))


def assert_card(uuid, startup=False):
    state = snapshot()
    gpu = next(g for g in state['devices'] if g['uuid'] == uuid)
    apps = [row.split(',') for row in state['compute_apps']]
    other = [row for row in apps if row[0].strip() == uuid and int(row[1]) != os.getpid()]
    if gpu['index'] != 3 or other or (startup and (gpu['free_mib'] < 19500 or gpu['utilization'] > 10)):
        raise RuntimeError(f'GPU 3 lacks required headroom: {gpu}, other processes: {other}')


def verify_small_mask():
    # Target IDs 0 and 1 have identical bytes. Each keeps its own actual GT;
    # the third query sees one copy of those bytes, never two negatives.
    labels = torch.tensor([0, 1, 2])
    mask = torch.tensor([[1,0,1,1], [0,1,1,1], [1,0,1,1]], dtype=torch.bool)
    ids, local, m = batch_columns(labels, mask, torch.tensor([0,0,2,3]))
    assert ids.tolist() == [0,1,2] and local.tolist() == [0,1,2]
    assert m.tolist() == [[True,False,True],[False,True,True],[True,False,True]]
    ids, local, m = batch_columns(labels[:2], mask[:2], torch.tensor([0,0,2,3]))
    assert ids.tolist() == [0,1] and m.sum(1).tolist() == [1,1]


def smoke(root, model, context, canonical):
    model.eval()
    b = context['banks']['fit']
    positions = b['indices'][:32]
    loss_fn = masked_batch_objective(canonical)
    params = [p for group in model.parameter_groups() for p in group['params']]
    optimizer = torch.optim.AdamW(model.parameter_groups())
    with torch.autocast('cuda', dtype=torch.bfloat16):
        parts = encode_parts(model, [context['rows'][i] for i in positions], ARM)
        out = fuse_parts(model,parts,context['reference'][positions],context['written'][positions],ARM)
        full = objective(out,b['gallery'],b['labels'][:32],b['mask'][:32],context['teacher'][positions],ARM)
        loss = loss_fn(out,b['gallery'],b['labels'][:32],b['mask'][:32],context['teacher'][positions],ARM)
    if not all(torch.isfinite(x['total']) for x in (full,loss)):
        raise FloatingPointError('Nonfinite B32 smoke loss')
    loss['total'].backward()
    torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
    optimizer.step()
    if (not torch.equal(model.embedding.weight[:49408], model.clip.token_embedding.weight)
            or any(p.grad is not None for p in model.clip.visual.parameters())):
        raise ValueError('B32 freeze smoke failed')
    write(root/'smoke.json',dict(status='passed',batch=32,optimizer_updates=1,discarded=True,
          full_loss=float(full['total'].detach()), inbatch_loss=float(loss['total'].detach()),
          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
          peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))


def run(root, category, uuid, smoke_only=False):
    directory = root / (category + ('_smoke' if smoke_only else ''))
    directory.mkdir()
    path = directory/'state.json'
    started = time.monotonic()
    state = dict(status='running', category=category, identity=identity(), gpu=uuid, api_calls=0)

    def progress(stage, done=0, total=0, **fields):
        if (root/'cancel.request').exists() or time.monotonic()-started > 7200:
            raise InterruptedError('Experiment cancelled or timed out')
        assert_card(uuid)
        state.update(stage=stage, done=done, total=total, **fields)
        write(path,state)

    try:
        assert_card(uuid,startup=True)
        require_space(root,20)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
            raise RuntimeError('GPU UUID was not bound')
        for name,digest in read(root/'plan.json')['source_sha256'].items():
            if sha(root/'code'/name) != digest:
                raise ValueError(f'Sealed source changed: {name}')
        verify_small_mask()
        progress('source_checks')
        torch.manual_seed(42)
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.75)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        meta, _, base, base_sha = load_sources(category)
        cached = cached_bank(category,'train',dict(base=base_sha,metadata=meta))
        context = training_context(cached,meta)
        del cached
        canonical = canonical_targets(meta)
        model, transform = make_model(base)
        initial = model.delta_state()
        if smoke_only:
            progress('batch32_smoke')
            smoke(directory,model,context,canonical)
            state.update(status='passed',stage='smoke_complete')
            write(path,state)
            return
        reports = {}
        for pool,batch in ARMS:
            name = f'{pool}_b{batch}'
            destination = directory/name
            destination.mkdir()
            progress('starting_fit',arm=name)
            model.load_delta(initial)
            torch.manual_seed(42)
            torch.cuda.reset_peak_memory_stats()
            loss = masked_batch_objective(canonical) if pool == 'inbatch' else None
            choice = fit(destination,model,context,lambda s,d,t:progress(s,d,t,arm=name),
                         batch_size=batch,train_objective=loss)
            r = read(destination/'fit_results.json')
            n = len(context['banks']['fit']['indices'])
            if r['updates'] != 3*math.ceil(n/batch) or r['exposures'] != [3]*n:
                raise ValueError(f'Unexpected exposure for {name}')
            reports[name] = dict(selected=choice,history=r['history'],updates=r['updates'],
                                 exposures=r['exposures'],negatives=negative_stats(context['banks']['fit'],canonical,pool,batch),
                                 peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                                 peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            write(directory/'fits_summary.json',reports)
        del context, canonical
        gc.collect()
        torch.cuda.empty_cache()
        model.load_delta(initial)
        model.eval()
        val, hashes, native = official_bank(category)
        rows = [dict(row,native_text=text) for row,text in zip(val['rows'],native)]
        bank = encode_bank(model.clip,transform,model.tokenizer,category,'val',rows,val['ids'],hashes,
                           lambda s,d,t:progress(s,d,t,arm='base_gallery'))
        official = {}

        def evaluate_arm(name, dqu=False):
            scores, _ = eval_global(model,bank,dqu,lambda d,t:progress('official',d,t,arm=name))
            for view,value in scores.items():
                ranks = ranks_for(value,bank)
                official[name if dqu else name+'_'+view] = dict(metrics(ranks),ranks=ranks)

        evaluate_arm('dqu',dqu=True)
        for name,report in reports.items():
            choice = report['selected']
            weight = directory/name/choice['weight']
            if sha(weight) != choice['sha256']:
                raise ValueError(f'Selected weight changed: {name}')
            model.load_delta(torch.load(weight,map_location='cpu',weights_only=True)['state'])
            evaluate_arm(name)
        contrasts = [('inbatch_b8','full_b8'),('inbatch_b32','full_b32'),
                     ('inbatch_b32','inbatch_b8'),('full_b32','full_b8')]
        pairs = [(a+'_'+view,b+'_'+view) for a,b in contrasts for view in ('raw','native')]
        write(directory/'results.json',dict(category=category,queries=len(rows),gallery=len(val['ids']),rows=official,
             fits=reports,api_calls=0,base_sha256=base_sha,comparisons={a+'_minus_'+b:{str(k):transitions(
                official[a]['ranks'],official[b]['ranks'],k) for k in (1,10,50)} for a,b in pairs}))
        state.update(status='complete',stage='complete',seconds=time.monotonic()-started)
        write(path,state)
    except Exception as error:
        state.update(status='failed',error=type(error).__name__,detail=str(error))
        write(path,state)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--category',choices=CATEGORIES,required=True)
    p.add_argument('--gpu',required=True)
    p.add_argument('--smoke',action='store_true')
    a = p.parse_args()
    run(a.root,a.category,a.gpu,a.smoke)
