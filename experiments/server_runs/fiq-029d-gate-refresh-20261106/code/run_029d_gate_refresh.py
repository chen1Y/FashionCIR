"""Matched 029D continuation on the Dress and Shoes gate ablation."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from attr_transfer_data import official_inputs
from fiq_native_caption_math_031 import metrics, transitions
from run_attr_transfer_029d import ARM, BASE_CACHE, encode_bank, fit, training_context
from run_dqu_caption_transfer_controls import CHECKPOINTS, EXPECTED
from run_fiq_native_caption_031 import BASE, identity, read, sha, snapshot, write


PARENT = BASE / 'runs/fiq-029d-dress-shoes-20261106'
CODE = Path('/home/lcy/fashionCIR/src')
ARMS = ('original', 'refreshed')
PAIRWISE = (('refreshed_native', 'original_native'), ('refreshed_raw', 'original_raw'),
            ('refreshed_native', 'refreshed_native_beta0'), ('refreshed_native', 'native_dqu'),
            ('refreshed_native', 'starting_native'))


def restore(model, delta, refreshed=False, widen=False):
    from fiq_029d_slot_refresh import ModificationAwareGate
    from fiq_gate_patch_model_028 import AttributeGate

    model.attribute_gate = AttributeGate().cuda()
    if refreshed and not widen:
        model.attribute_gate = ModificationAwareGate(model.attribute_gate).cuda()
    model.load_delta(delta)
    if refreshed and widen:
        model.attribute_gate = ModificationAwareGate(model.attribute_gate).cuda()
    model.strict_segment_isolation = refreshed
    model.configure_arm(ARM)


def dev_metrics(model, context, split, arm):
    import torch
    from expanded_attr_model import ranks_from_scores
    from fiq_029d_slot_refresh import compose_refreshed
    from fiq_fusion_029 import encode_parts

    bank = context['banks'][split]
    ranks, nce = [], []
    model.eval()
    with torch.no_grad():
        for j in range(0, len(bank['indices']), 8):
            ix = bank['indices'][j:j+8]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                parts = encode_parts(model, [context['rows'][i] for i in ix], arm)
                out = compose_refreshed(model, parts, context['reference'][ix], context['written'][ix], arm)
            scores = out['q'].float() @ bank['gallery'].T
            labels = bank['labels'][j:j+len(ix)]
            refs = bank['references'][j:j+len(ix)]
            ranks.extend(ranks_from_scores(scores, labels, refs).tolist())
            logits = scores.masked_fill(~bank['mask'][j:j+len(ix)], -torch.inf)
            nce.extend(torch.nn.functional.cross_entropy(logits / .07, labels, reduction='none').tolist())
    return dict(metrics(ranks), ranks=ranks, retrieval_nce=sum(nce)/len(nce))


def evaluate_official(category, model, bank, base, start, selected, root, progress):
    import torch
    import torch.nn.functional as F
    from expanded_attr_model import ranks_from_scores
    from fiq_029d_slot_refresh import compose_refreshed
    from fiq_fusion_029 import encode_parts, fuse_parts
    from train_clean_attr_alignment import plain_text

    rows = bank['rows']
    ranks = {}
    gate = {}
    reference, written, gallery = bank['reference'], bank['written'], bank['gallery']

    def collect(name, query, j, weights=None, valid=None):
        stop = j + len(query)
        scores = query.float() @ gallery.T
        if not torch.isfinite(scores).all():
            raise FloatingPointError('Nonfinite official score')
        labels, refs = bank['labels'][j:stop], bank['references'][j:stop]
        keep = labels != refs
        output = torch.full_like(labels, len(gallery))
        output[keep] = ranks_from_scores(scores[keep], labels[keep], refs[keep])
        ranks.setdefault(name, []).extend(output.tolist())
        if weights is not None:
            multi = valid.sum(1) >= 2
            w = weights[multi]
            counts = valid[multi].sum(1).float()
            stat = gate.setdefault(name, dict(multi=0, max_sum=0., entropy_sum=0., uniform_l1_sum=0.))
            stat['multi'] += len(w)
            stat['max_sum'] += float(w.max(1).values.sum())
            stat['entropy_sum'] += float((-(w * w.clamp_min(1e-12).log()).sum(1) / counts.log()).sum())
            stat['uniform_l1_sum'] += float((w - valid[multi]/counts[:,None]).abs().sum())

    restore(model, base)
    model.eval()
    with torch.inference_mode():
        for j in range(0, len(rows), 8):
            batch = rows[j:j+8]
            w = written[j:j+len(batch)]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                t = F.normalize(model.clip.encode_text(model.tokenizer([r['native_text'] for r in batch]).cuda()).float(), dim=-1)
                q, _ = model.image_fusion(t, w)
            collect('native_dqu', q, j)
    for name in ('starting', *ARMS):
        refreshed = name == 'refreshed'
        if name == 'starting':
            delta = start
        else:
            checkpoint = root / name / selected[name]['weight']
            if sha(checkpoint) != selected[name]['sha256']:
                raise ValueError('Selected checkpoint checksum mismatch')
            delta = torch.load(checkpoint, map_location='cpu', weights_only=True)['state']
        restore(model, delta, refreshed)
        model.eval()
        compose = compose_refreshed if refreshed else fuse_parts
        with torch.inference_mode():
            for j in range(0, len(rows), 8):
                batch = rows[j:j+8]
                w, v = written[j:j+len(batch)], reference[j:j+len(batch)]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    parts = encode_parts(model, batch, ARM)
                    raw = compose(model, parts, v, w, ARM)
                    m = plain_text(model, [row['native_text'] for row in batch])
                    native = compose(model, dict(parts, mod=m), v, w, ARM)
                    q0, _ = model.image_fusion(m, w)
                collect(name+'_raw', raw['q'], j, raw['slot_weights'], parts['valid'])
                collect(name+'_native', native['q'], j, native['slot_weights'], parts['valid'])
                collect(name+'_native_beta0', q0, j)
                if j % 128 == 0:
                    progress('official_'+name, j+len(batch), len(rows))
    stats = {name:dict(multi_slot_queries=d['multi'], mean_max_weight=d['max_sum']/d['multi'],
                       normalized_entropy=d['entropy_sum']/d['multi'],
                       mean_l1_from_uniform=d['uniform_l1_sum']/d['multi']) for name,d in gate.items() if d['multi']}
    comparisons = {a+'_minus_'+b: {str(k):transitions(ranks[a], ranks[b], k) for k in (1,10,50)} for a,b in PAIRWISE}
    previous = read(PARENT/category/'official_results.json')['rows']
    identity = {a:ranks[a]==previous[b]['ranks'] for a,b in (
        ('native_dqu','native_dqu_caption_corrected'),('starting_raw','029d_raw_original'),
        ('starting_native','029d_native_input'),('starting_native_beta0','029d_native_input_beta0'))}
    write(root/'official_results.json', dict(category=category, queries=len(rows), gallery=len(gallery),
          rows={name:dict(metrics(r),ranks=r) for name,r in ranks.items()}, gate_stats=stats,
          comparisons=comparisons, original_rank_reproduction=identity,
          reference_target_collisions=int((bank['labels']==bank['references']).sum())))


def run(root, category):
    path = root/'state.json'
    if path.exists():
        raise FileExistsError(path)
    start = time.monotonic()
    state = dict(status='running', phase='loading', identity=identity(), category=category, api_calls=0)
    write(path,state)

    def progress(phase, completed, total):
        if time.monotonic()-start > 3600:
            raise TimeoutError('One-hour category deadline')
        if (root.parent/'cancel.request').exists():
            raise InterruptedError('Cancelled')
        state.update(phase=phase,completed=completed,total=total)
        write(path,state)

    try:
        import cv2
        import torch
        from fiq_029d_slot_refresh import compose_refreshed
        from fiq_fusion_029 import FusionText029
        from train_attr_slot_pilot import load_clip

        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.cuda.set_per_process_memory_fraction(.4)
        source = PARENT/category
        meta_path = source/'train.json'
        if sha(meta_path)!=read(PARENT/'launch.json')['inputs'][category]:
            raise ValueError('Training metadata differs from original experiment')
        meta = read(meta_path)
        prior = read(source/'fit_results.json')['selected']
        weights = source/prior['weight']
        if sha(weights)!=prior['sha256']:
            raise ValueError('Starting 029D checkpoint checksum mismatch')
        delta = torch.load(weights,map_location='cpu',weights_only=True)['state']
        checkpoint = BASE_CACHE/category/'base.pt'
        if sha(checkpoint)!=CHECKPOINTS[category][1]:
            raise ValueError('Base checkpoint checksum mismatch')
        clip,transform,tokenizer = load_clip(checkpoint)
        bank=encode_bank(clip,transform,tokenizer,category,'train',meta['records'],
                         meta['banks']['fit']+meta['banks']['dev'],meta['image_sha256'],progress)
        ctx=training_context(bank,meta)
        model=FusionText029(clip,tokenizer,2).cuda()
        model.image_fusion.from_dqu(checkpoint)
        model.configure_arm(ARM)
        base=model.delta_state()
        selected={}
        for arm in ARMS:
            folder=root/arm
            folder.mkdir()
            refreshed=arm=='refreshed'
            restore(model,delta,refreshed,widen=refreshed)
            torch.manual_seed(42)
            report=lambda phase,completed,total: progress(arm+'_'+phase,completed,total)
            report('fit_smoke',0,1)
            selected[arm]=fit(folder,model,ctx,report,compose=compose_refreshed if refreshed else None,
                              score_dev=dev_metrics if refreshed else None)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        del ctx,bank
        rows,ids,hashes,text_sha=official_inputs(category)
        if (len(rows),len(ids))!=EXPECTED[category]:
            raise ValueError('Official evaluation denominator changed')
        bank=encode_bank(clip,transform,tokenizer,category,'val',rows,ids,hashes,progress)
        evaluate_official(category,model,bank,base,delta,selected,root,progress)
        state.update(status='complete',phase='finished',completed=len(rows),total=len(rows),
                     seconds=time.monotonic()-start,starting_sha256=prior['sha256'],source_text_sha256=text_sha,
                     selected_epochs={a:h['epoch'] for a,h in selected.items()},
                     peak_allocated_gib=torch.cuda.max_memory_allocated()/(1<<30))
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write(path,state)


def launch(root):
    if (root/'launch.json').exists() or any((root/c).exists() for c in EXPECTED):
        raise FileExistsError('Existing run')
    first=snapshot()
    time.sleep(5)
    second=snapshot()
    cards=[next(d for d in s['devices'] if d['index']==3) for s in (first,second)]
    uid=cards[0]['uuid']
    if (any(d['uuid']!=uid or d['free_mib']<20480 or d['utilization']>10 for d in cards)
            or any(line.startswith(uid+',') for s in (first,second) for line in s['compute_apps'])):
        raise RuntimeError('GPU 3 did not pass the two-process resource check')
    env={name:os.environ[name] for name in ('HOME','PATH','LANG') if name in os.environ}
    env.update(CUDA_VISIBLE_DEVICES=uid,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH=str(CODE))
    workers={}
    for category in EXPECTED:
        folder=root/category
        folder.mkdir()
        with (folder/'run.log').open('x') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(folder),'--category',category],
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                   env=env,start_new_session=True)
        workers[category]=child.pid
    write(root/'launch.json',dict(workers=workers,gpu=cards[1],snapshots=[first,second],api_calls=0,
                                sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}))
    print(json.dumps(workers),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--category',choices=EXPECTED)
    args=parser.parse_args()
    if args.category:
        run(args.root,args.category)
    else:
        launch(args.root)
