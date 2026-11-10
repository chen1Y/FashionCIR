"""Train 029D with both CLIP towers on the existing FashionIQ fit/dev splits."""

import argparse
import gc
import hashlib
import io
import math
import os
import time
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F

from attr_transfer_data import official_inputs
from expanded_attr_model import legal_masks, ranks_from_scores
from experiment_storage import atomic_torch_save, require_space
from fiq_fusion_029 import FusionText029, encode_parts, fuse_parts
from fiq_native_caption_math_031 import metrics, transitions
from run_029d_gate_refresh import ARM, SHIRT_TOPTEE, inputs_029, load_training_sources
from run_029d_spelling_pair import cached_bank, text_manifest, text_views
from run_attr_transfer_029d import load_dataset
from run_dqu_caption_transfer_controls import CHECKPOINTS
from run_fiq_native_caption_031 import BASE, identity, read, resolve, sha, snapshot, write
from train_attr_slot_pilot import load_clip
from train_clean_attr_alignment import plain_text


CATEGORIES = ('dress', 'shirt', 'toptee')
SEED = 42
BATCH = 8
EPOCHS = 3
DEVICE_INDEX = '3'


def assert_gpu(uuid, starting=False):
    view = snapshot()
    card = next(g for g in view['devices'] if g['uuid'] == uuid)
    if str(card['index']) != DEVICE_INDEX:
        raise RuntimeError('The requested GPU UUID is no longer physical GPU 3')
    jobs = [line.split(',') for line in view['compute_apps']]
    others = [j for j in jobs if j[0].strip() == uuid and int(j[1]) != os.getpid()]
    if others or (starting and (card['free_mib'] < 22528 or card['utilization'] > 10)):
        raise RuntimeError(f'GPU 3 lacks exclusive headroom: {card}, other processes: {others}')
    return card


class FullEmbedding(torch.nn.Module):
    """Share the original vocabulary with CLIP and keep new-token LR separate."""

    def __init__(self, base, extra):
        super().__init__()
        self.base = base
        self.extra = torch.nn.Parameter(extra.detach().clone())

    @property
    def weight(self):
        return self.base.weight

    def forward(self, tokens):
        return F.embedding(tokens, torch.cat((self.base.weight, self.extra)))


def bind_full_clip(model):
    model.embedding = FullEmbedding(model.clip.token_embedding, model.embedding.extra)
    model.clip.requires_grad_(True)
    # The experiment uses the existing fixed 0.07 temperature, outside CLIP's forward.
    model.clip.logit_scale.requires_grad_(False)
    if getattr(model.clip, 'logit_bias', None) is not None:
        model.clip.logit_bias.requires_grad_(False)
    model.clip.set_grad_checkpointing(True)


def image_inputs(category, transform, hashes):
    path_for, render = load_dataset(category, transform)

    @lru_cache(maxsize=None)
    def verify(image_id):
        path = path_for(image_id)
        if sha(path) != hashes[image_id]:
            raise ValueError(f'Image checksum mismatch: {image_id}')

    def image(image_id):
        path = path_for(image_id)
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != hashes[image_id]:
            raise ValueError(f'Image checksum mismatch: {image_id}')
        with Image.open(io.BytesIO(blob)) as original:
            return transform(original.convert('RGB'))

    @lru_cache(maxsize=1200)
    def written(reference, target):
        verify(reference)
        return render({'candidate': reference, 'target': target}, 'val')

    return image, written


def vision(clip, images):
    return F.normalize(clip.encode_image(torch.stack(images).cuda()).float(), dim=-1)


@torch.no_grad()
def gallery_features(clip, ids, image, progress):
    result = []
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(ids), BATCH):
            result.append(vision(clip, [image(k) for k in ids[i:i+BATCH]]))
            if i % 512 == 0:
                progress(i + len(result[-1]), len(ids))
    result = torch.cat(result)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite gallery feature')
    return result


def checked_bank(meta, split):
    ids = meta['banks'][split]
    mask, labels = legal_masks(meta, split)
    rows = [meta['records'][i] for i in meta['offsets'][split]]
    return dict(ids=ids, rows=rows, labels=labels.cuda(), mask=mask.cuda())


def backward_batch(model, bank, positions, negatives, teacher, image, written):
    rows = [bank['rows'][i] for i in positions.tolist()]
    pos = positions.cuda()
    labels = bank['labels'][pos]
    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        reference = vision(model.clip, [image(r['candidate']) for r in rows])
        painted = vision(model.clip, [written(r['candidate'], r['target']) for r in rows])
        target = vision(model.clip, [image(r['target']) for r in rows])
        parts = encode_parts(model, rows, ARM)
        out = fuse_parts(model, parts, reference, painted, ARM)
    # All legal negatives come from the detached epoch snapshot. Replacing each
    # positive with its live feature gives the target branch a direct gradient.
    query = out['q'].float()
    scores = (query @ negatives.T).scatter(1, labels[:, None], (query * target).sum(1, keepdim=True))
    ret = F.cross_entropy(scores.masked_fill(~bank['mask'][pos], -torch.inf)/.07, labels)
    valid = out['valid']
    good = valid.any(1)
    glob = ((1 - (out['global_text'] * target).sum(1)) * good).sum() / good.sum().clamp_min(1)
    slot = ((1 - (out['slots'] * teacher[pos]).sum(-1)) * valid).sum() / valid.sum().clamp_min(1)
    loss = ret + .1 * glob + .1 * slot
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite full-CLIP loss')
    loss.backward()
    return float(loss.detach())


@torch.no_grad()
def evaluate(model, bank, gallery, written, progress, *, baseline=False, native_text=None):
    model.eval()
    positions = {key:i for i,key in enumerate(bank['ids'])}
    refs = torch.tensor([positions[r['candidate']] for r in bank['rows']], device='cuda')
    labels, mask = bank['labels'], bank.get('mask')
    names = ('native_dqu',) if baseline else ('raw', 'native') if native_text is not None else ('raw',)
    ranks = {name:[] for name in names}
    nce = []
    for i in range(0, len(refs), BATCH):
        rows = bank['rows'][i:i+BATCH]
        ix = slice(i, i+len(rows))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            painted = vision(model.clip, [written(r['candidate'], r['target']) for r in rows])
            if baseline:
                text = F.normalize(model.clip.encode_text(model.tokenizer(native_text[ix]).cuda()).float(), dim=-1)
                queries = {'native_dqu': model.image_fusion(text, painted)[0]}
            else:
                parts = encode_parts(model, rows, ARM)
                reference = gallery[refs[ix]]
                queries = {'raw': fuse_parts(model, parts, reference, painted, ARM)['q']}
                if native_text is not None:
                    queries['native'] = fuse_parts(model, dict(parts, mod=plain_text(model,native_text[ix])),
                                                   reference, painted, ARM)['q']
        for name, query in queries.items():
            scores = query.float() @ gallery.T
            ranks[name].extend(ranks_from_scores(scores, labels[ix], refs[ix]).tolist())
            if mask is not None:
                nce.extend(F.cross_entropy(scores.masked_fill(~mask[ix], -torch.inf)/.07,
                                           labels[ix], reduction='none').tolist())
        if i % 128 == 0:
            progress(i + len(rows), len(refs))
    return {name:dict(metrics(r), ranks=r) for name,r in ranks.items()}, sum(nce)/max(1,len(nce))


def load_sources(category):
    meta, selected, selected_sha, base, base_sha = load_training_sources(category)
    if not base.is_file():
        base = (BASE / 'autodl-tmp' / CHECKPOINTS[category][0] if category == 'dress' else
                resolve(f'/root/autodl-tmp/results/fiq_shirt_toptee_20261022/{category}/checkpoints/{category}_dqu_original_seed42_best.pt'))
    if sha(base) != base_sha or sha(selected) != selected_sha:
        raise ValueError('Base or 029D reference weights differ from their sealed SHA')
    return meta, selected, base, base_sha


def official_bank(category):
    if category in SHIRT_TOPTEE:
        meta, _ = inputs_029(category, 'val')
        rows, ids, hashes = meta['records'], meta['gallery_ids'], meta['image_sha256']
    else:
        rows, ids, hashes, _ = official_inputs(category)
    _, native = text_views(category, 'val', rows)
    positions = {name:i for i,name in enumerate(ids)}
    labels = torch.tensor([positions[r['target']] for r in rows], device='cuda')
    expected = {'dress':(2017,3817), 'shirt':(2038,6346), 'toptee':(1961,5373)}
    if (len(rows),len(ids)) != expected[category] or len(positions) != len(ids):
        raise ValueError('Official gallery/query count changed')
    return dict(rows=rows, ids=ids, labels=labels), hashes, native['official_dict']


def make_model(base):
    clip, transform, tokenizer = load_clip(base)
    model = FusionText029(clip, tokenizer, 2).cuda()
    model.image_fusion.from_dqu(base)
    model.configure_arm(ARM)
    return model, transform


def run(root, category, uuid, *, smoke=False):
    directory = root / (category + ('_smoke' if smoke else ''))
    directory.mkdir()
    status = directory/'state.json'
    state = dict(status='running', identity=identity(), category=category, gpu=uuid, api_calls=0)
    started = time.monotonic()

    def progress(stage, done=0, total=0, **details):
        if (root/'cancel.request').exists() or time.monotonic()-started > 14400:
            raise InterruptedError('Run cancellation or category timeout')
        assert_gpu(uuid)
        if stage in ('source_checks','save_best'):
            require_space(root, 30)
        state.update(stage=stage, done=done, total=total, **details)
        write(status,state)

    try:
        assert_gpu(uuid,starting=True)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
            raise RuntimeError('GPU UUID must be bound explicitly')
        plan = read(root/'plan.json')
        if sha(Path(__file__).resolve()) != plan['source_sha256']:
            raise ValueError('Frozen experiment source changed')
        progress('source_checks')
        torch.manual_seed(SEED)
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.92)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        meta, selected, base, base_sha = load_sources(category)
        manifest = text_manifest(category)
        initial = cached_bank(category, 'train', dict(base=base_sha, metadata=meta))
        bank, dev = [checked_bank(meta,s) for s in ('fit','dev')]
        teacher = initial['teacher'][meta['offsets']['fit']].cuda()
        base_gallery = initial['gallery'][[initial['positions'][key] for key in bank['ids']]].cuda()
        del initial
        model, transform = make_model(base)
        image, written = image_inputs(category,transform,meta['image_sha256'])
        image_fit = lru_cache(maxsize=1800)(image)
        bind_full_clip(model)
        last2 = list(model.clip.transformer.resblocks[-2:].parameters())
        last2_ids = {id(p) for p in last2}
        other_clip = [p for p in model.clip.parameters() if p.requires_grad and id(p) not in last2_ids]
        groups = [dict(params=other_clip,lr=1e-6,weight_decay=.01),
                  dict(params=last2,lr=1e-5,weight_decay=.01),
                  dict(params=[model.embedding.extra],lr=2e-4,weight_decay=0.),
                  dict(params=list(model.image_fusion.parameters()),lr=1e-4,weight_decay=.01),
                  dict(params=list(model.attribute_gate.parameters())+list(model.text_mixture.parameters()),lr=2e-4,weight_decay=.01)]
        params = [p for group in groups for p in group['params']]
        optimizer = torch.optim.AdamW(groups, fused=True)
        counts = dict(vision=sum(p.numel() for p in model.clip.visual.parameters()),
                      clip=sum(p.numel() for p in other_clip+last2), total=sum(p.numel() for p in params))
        rng = torch.Generator().manual_seed(SEED)
        updates = 0
        exposure = torch.zeros(len(bank['rows']),dtype=torch.int64)
        history = []
        best = None
        weight = directory/'selected.pt'

        def dev_checkpoint(epoch):
            gallery = gallery_features(model.clip,dev['ids'],image,lambda d,t:progress('dev_gallery',d,t,epoch=epoch))
            result,nce = evaluate(model,dev,gallery,written,lambda d,t:progress('dev_ranks',d,t,epoch=epoch))
            record = dict(epoch=epoch, updates=updates, nce=nce, **result['raw'])
            history.append(record)
            return record

        def step(positions):
            nonlocal updates
            optimizer.zero_grad(set_to_none=True)
            loss = backward_batch(model,bank,positions,negatives,teacher,image_fit,written)
            norm = torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            return loss, float(norm)

        def mark_best(record):
            nonlocal best
            key = lambda h: (h['r10']+h['r50'],-h['nce'],-h['epoch'])
            if best is None or key(record)>key(best):
                progress('save_best',epoch=record['epoch'])
                atomic_torch_save(dict(epoch=record['epoch'], state=model.state_dict()),weight)
                best = record
            write(directory/'fit_history.json',dict(history=history,selected_epoch=best['epoch'],parameters=counts))

        negatives = base_gallery
        if smoke:
            model.train()
            progress('first_optimizer_step')
            before = model.clip.visual.conv1.weight.detach().clone()
            old_token = model.embedding.weight[1].detach().clone()
            positions = torch.randperm(len(exposure),generator=rng)[:BATCH]
            loss,norm = step(positions)
            old_vocab_grad = model.embedding.weight.grad
            changed = (float((model.clip.visual.conv1.weight-before).abs().max()),
                       float(old_vocab_grad.norm()),float((model.embedding.weight[1]-old_token).abs().max()))
            if not all(math.isfinite(v) for v in changed) or min(changed[:2]) <= 0:
                raise ValueError('Full-CLIP gradient/parameter-change probe failed')
            result = dict(status='passed', updates=updates, parameters=counts, loss=loss, gradient_norm=norm,
                          vision_conv_change=changed[0], original_vocab_gradient_norm=changed[1],
                          original_vocab_row1_change=changed[2],
                          peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            write(directory/'smoke.json',result)
            state.update(result,stage='smoke_complete')
            write(status,state)
            return

        model.eval()
        mark_best(dev_checkpoint(0))
        for epoch in range(1,EPOCHS+1):
            if epoch>1:
                negatives = gallery_features(model.clip,bank['ids'],image,lambda d,t:progress('fit_gallery_refresh',d,t,epoch=epoch))
            model.train()
            order = torch.randperm(len(exposure),generator=rng)
            for i in range(0,len(order),BATCH):
                pos = order[i:i+BATCH]
                loss,norm = step(pos)
                exposure[pos] += 1
                if updates % 16 == 0 or i==0:
                    progress('fit',i+len(pos),len(order),epoch=epoch,updates=updates,loss=loss,gradient_norm=norm)
            optimizer.zero_grad(set_to_none=True)
            mark_best(dev_checkpoint(epoch))
        if not bool((exposure==EPOCHS).all()):
            raise ValueError('Unexpected number of presentations')
        del optimizer, params, groups, negatives, base_gallery, teacher, bank, dev
        gc.collect()
        torch.cuda.empty_cache()
        best_sha = sha(weight)
        saved = torch.load(weight,map_location='cpu',weights_only=True,mmap=True)
        model.load_state_dict(saved['state'],strict=True)
        del saved
        model.eval()
        full_fit = dict(history=history, selected_epoch=best['epoch'], selected_sha256=best_sha,
                        updates=updates, exposure=exposure.tolist(), parameters=counts,
                        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        write(directory/'fit_complete.json',full_fit)
        val,hashes,native_text = official_bank(category)
        val_image,val_written = image_inputs(category,transform,hashes)
        gallery = gallery_features(model.clip,val['ids'],val_image,lambda d,t:progress('official_full_gallery',d,t))
        raw, _ = evaluate(model,val,gallery,val_written,lambda d,t:progress('official_full_ranks',d,t),native_text=native_text)
        del model,gallery
        gc.collect()
        torch.cuda.empty_cache()
        baseline,_ = make_model(base)
        baseline.eval()
        gallery = gallery_features(baseline.clip,val['ids'],val_image,lambda d,t:progress('official_base_gallery',d,t))
        dqu,_ = evaluate(baseline,val,gallery,val_written,lambda d,t:progress('official_dqu_ranks',d,t),
                         baseline=True,native_text=native_text)
        baseline.load_delta(torch.load(selected,map_location='cpu',weights_only=True)['state'])
        old,_ = evaluate(baseline,val,gallery,val_written,lambda d,t:progress('official_last2_ranks',d,t),native_text=native_text)
        rows = {**dqu, **{f'full_clip_{k}':v for k,v in raw.items()}, **{f'last2_{k}':v for k,v in old.items()}}
        if manifest != text_manifest(category) or any(len(v['ranks'])!=len(val['rows']) for v in rows.values()):
            raise ValueError('Official data or denominator changed')
        pairs = [(f'full_clip_{k}',f'last2_{k}') for k in ('raw','native')] + [('full_clip_native','native_dqu')]
        write(directory/'results.json',dict(category=category,queries=len(val['rows']),gallery=len(val['ids']),
              rows=rows,comparisons={f'{a}_minus_{b}':{str(k):transitions(rows[a]['ranks'],rows[b]['ranks'],k)
                     for k in (1,10,50)} for a,b in pairs},fit=full_fit,base_sha256=base_sha,api_calls=0))
        state.update(status='complete',stage='complete',seconds=time.monotonic()-started)
        write(status,state)
    except Exception as err:
        state.update(status='failed',error=type(err).__name__,detail=str(err))
        write(status,state)
        raise


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--category',choices=CATEGORIES,required=True)
    p.add_argument('--gpu',required=True)
    p.add_argument('--smoke',action='store_true')
    a = p.parse_args()
    run(a.root,a.category,a.gpu,smoke=a.smoke)
