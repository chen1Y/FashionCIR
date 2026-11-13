"""039A/B on the 038G joint FashionIQ split, with a disk-backed ViT prefix cache."""

import argparse
import gc
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import parametrize

from experiment_storage import atomic_torch_save, require_space
from fiq_fusion_029 import encode_parts
from fiq_native_caption_math_031 import transitions
from fiq_pixel_free_038 import PixelFreeVisual, compose, loss_for
from fiq_vit_lora_039 import attach_lora, live_gallery_loss, tail_features, tail_outputs
from run_029d_freeze_patch import ranks_for
from run_029d_openclip_036 import from_public_clip, gpu_snapshot, official_bank
from run_attr_transfer_029d import encode_bank
from run_fiq_native_caption_031 import identity, read, sha, write
from run_fiq_pixel_free_038 import (
    ARM, BATCH, CATEGORIES, COUNTS, SEED, bind_drops, drop_features, full_fit_bank, make_bank, official_result,
)
from train_attr_pair_alignment import frozen_digest


EPOCHS = 3
REFRESH_STEPS = 50
TAIL_BATCH = 32
PREFIX_SHAPE = (257, 1280)


class CapacityError(RuntimeError):
    """The assigned GPU no longer has room for the registered allocation budget."""


def cache_bank(model, transform, category, split, rows, ids, hashes, path, progress):
    """Capture one exact BF16 sequence per JPEG during ordinary frozen encoding."""
    stored = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint16, shape=(len(ids), *PREFIX_SHAPE))
    offset = 0
    first = None

    def capture(_module, _inputs, hidden):
        nonlocal offset, first
        sequence = hidden.transpose(0, 1).detach()
        if sequence.dtype != torch.bfloat16 or tuple(sequence.shape[1:]) != PREFIX_SHAPE:
            raise ValueError(f'039 prefix layout changed: {sequence.shape}, {sequence.dtype}')
        n = len(sequence)
        if offset == 0:
            first = sequence.contiguous()
        stored[offset:offset + n] = sequence.contiguous().cpu().view(torch.uint16).numpy()
        offset += n

    def verify_first(_module, _inputs, output):
        nonlocal first
        if first is None:
            return
        feature = output[0] if isinstance(output, tuple) else output
        expected = torch.nn.functional.normalize(feature.float(), dim=-1)
        discrepancy = float((tail_features(model.clip.visual, first) - expected).abs().max())
        if not math.isfinite(discrepancy) or discrepancy > 1e-4:
            raise ValueError(f'039 first JPEG batch did not reproduce frozen features: {discrepancy:g}')
        write(path.with_suffix('.first_batch.json'), dict(rows=len(first), shape=PREFIX_SHAPE, max_abs=discrepancy))
        first = None

    handle = model.clip.visual.transformer.resblocks[-3].register_forward_hook(capture)
    verify_handle = model.clip.visual.register_forward_hook(verify_first)
    try:
        bank = make_bank(model, transform, category, split, rows, ids, hashes, False, progress)
    finally:
        handle.remove()
        verify_handle.remove()
    if offset != len(ids):
        raise ValueError('039 prefix capture count differs from gallery size')
    stored.flush()
    bank['prefix'] = stored
    return bank


def hidden_batch(cache, positions):
    array = np.array(cache['prefix'][positions], copy=True)
    return torch.from_numpy(array).view(torch.bfloat16).cuda()


def replay_bank(model, transform, category, rows, ids, hashes, source, progress, *, split='train',
                include_teacher=True):
    path = Path(source['path'])
    if path.stat().st_size != source['bytes'] or sha(path) != source['sha256']:
        raise ValueError(f'039 source prefix changed: {category}')
    prefix = np.load(path, mmap_mode='r', allow_pickle=False)
    if prefix.dtype != np.uint16 or prefix.shape != (len(ids), *PREFIX_SHAPE):
        raise ValueError(f'039 source prefix layout changed: {category}')

    def continue_batch(start, count):
        hidden = hidden_batch({'prefix': prefix}, slice(start, start + count))
        return tail_outputs(model.clip.visual, hidden)

    kwargs = {} if include_teacher else {'include_teacher': False}
    bank = encode_bank(model.clip, transform, model.tokenizer, category, split, rows, ids, hashes,
                       progress, include_written=False, cached_batch=continue_batch, **kwargs)
    bank['drop'], bank['drop_active'] = drop_features(model, rows)
    bank['prefix'] = prefix
    return bank


def check_split(vision, cache):
    n = min(8, len(cache['gallery']))
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        continued = tail_features(vision, hidden_batch(cache, slice(0, n)))
    discrepancy = float((continued - cache['gallery'][:n]).abs().max())
    if not math.isfinite(discrepancy) or discrepancy > 1e-4:
        raise ValueError(f'039 split did not reproduce the same-size frozen forward: {discrepancy:g}')
    return discrepancy


def score_bank(vision, cache, progress, indices=None):
    count = len(cache['gallery']) if indices is None else len(indices)
    outputs = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16), parametrize.cached():
        for start in range(0, count, TAIL_BATCH):
            stop = min(start + TAIL_BATCH, count)
            positions = slice(start, stop) if indices is None else indices[start:stop]
            outputs.append(tail_features(vision, hidden_batch(cache, positions)))
            if start % 512 == 0 or stop == count:
                progress(stop, count)
    result = torch.cat(outputs)
    if not torch.isfinite(result).all():
        raise FloatingPointError('039 nonfinite gallery feature')
    return result


def refresh_training(vision, training, update, progress):
    started = time.monotonic()
    for category, (cache, bank) in training.items():
        def report(done, total):
            progress('refresh_gallery', done, total, category=category, refresh_after_update=update)

        bank['gallery'] = score_bank(vision, cache, report)
    return round(time.monotonic() - started, 3)


def fit(directory, arm, model, visual, adapters, prepared, progress, *, query_builder=None, auxiliary=None,
        after_epoch=None, objective=None):
    directory.mkdir()
    training = {cat: (cache, dict(bank)) for cat, (cache, bank) in prepared.items()}
    lora_params = [p for adapter in adapters for p in adapter.parameters()]
    lora_ids = {id(p) for p in lora_params}
    groups = model.parameter_groups()
    for group in groups:
        group['params'] = [p for p in group['params'] if id(p) not in lora_ids]
    groups.append(dict(params=lora_params, lr=1e-4, weight_decay=.01))
    groups.append(dict(params=list(visual.parameters()), lr=1e-4, weight_decay=.01))
    params = [p for group in groups for p in group['params']]
    optimizer = torch.optim.AdamW(groups)
    frozen = frozen_digest(model.clip)
    shuffle = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)
    exposure = {cat: torch.zeros(len(bank['indices']), dtype=torch.int64) for cat, (_, bank) in training.items()}
    per_epoch = sum(math.ceil(len(counts) / BATCH) for counts in exposure.values())
    update = 0
    histories, refreshes = [], []

    def report(*args, **kwargs):
        try:
            progress(*args, **kwargs)
        except CapacityError:
            atomic_torch_save(dict(state=model.delta_state(), visual=visual.state_dict(),
                                   optimizer=optimizer.state_dict(), update=update, epoch=epoch,
                                   shuffle=shuffle.get_state(), exposure=exposure, history=histories,
                                   refreshes=refreshes, batches=batches, epoch_totals=dict(totals),
                                   cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                                   gallery={cat: bank['gallery'].cpu() for cat, (_, bank) in training.items()}
                                   if arm == '039B' else None), directory / 'interrupted_training.pt')
            raise

    epoch = 0
    while after_epoch is not None or epoch < EPOCHS:
        epoch += 1
        model.train()
        visual.train()
        batches = [(cat, ix) for cat in training
                   for ix in torch.randperm(len(exposure[cat]), generator=shuffle).split(BATCH)]
        batches = [batches[i] for i in torch.randperm(len(batches), generator=shuffle).tolist()]
        totals = Counter()
        for category, local in batches:
            if arm == '039B' and update and update % REFRESH_STEPS == 0:
                seconds = refresh_training(model.clip.visual, training, update, report)
                refreshes.append(dict(after_update=update, seconds=seconds))
                write(directory / 'refresh_history.json', refreshes)
            cache, bank = training[category]
            positions = [bank['indices'][i] for i in local.tolist()]
            refs = cache['references'][positions].cpu().numpy()
            labels = bank['labels'][local]
            mask = bank['mask'][local]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                reference = tail_features(model.clip.visual, hidden_batch(cache, refs), checkpoint_blocks=True)
                rows = [cache['rows'][i] for i in positions]
                drop, drop_active = cache['drop'][positions], cache['drop_active'][positions]
                if query_builder is None:
                    parts = encode_parts(model, rows, ARM)
                    out = compose(model, visual, parts, reference, drop, drop_active, None, ARM)
                else:
                    out = query_builder(model, visual, rows, reference, drop, drop_active)
                if objective is not None:
                    loss = objective(out, bank['gallery'], labels, mask)
                elif arm == '039B':
                    positive = tail_features(model.clip.visual, hidden_batch(cache, labels.cpu().numpy()), checkpoint_blocks=True)
                    if update == 0:
                        positive.retain_grad()
                    loss = live_gallery_loss(out, bank['gallery'], labels, mask, cache['teacher'][positions], positive)
                else:
                    loss = loss_for(out, bank['gallery'], labels, mask, cache['teacher'][positions], reference, ARM,
                                    lambda_rec=0., lambda_kl=0., reference_weight=0., lambda_gallery_kl=.5)
                if auxiliary is not None:
                    loss['opr'] = auxiliary(model, rows, bank['gallery'][labels])
                    loss['total'] = loss['total'] + .1 * loss['opr']
            if not all(torch.isfinite(value) for value in loss.values()):
                raise FloatingPointError('Nonfinite 039 objective')
            loss['total'].backward()
            if update == 0:
                if any(adapter.b.grad is None or not adapter.b.grad.count_nonzero() for adapter in adapters):
                    raise ValueError('039 first batch did not produce both ViT LoRA gradients')
                if arm == '039B' and (positive.grad is None or not positive.grad.count_nonzero()):
                    raise ValueError('039B first batch did not produce live positive gradients')
                if hasattr(model, 'gradient_audit'):
                    write(directory / 'gradient_preflight.json', model.gradient_audit())
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            if hasattr(model, 'embedding') and not torch.equal(
                    model.embedding.weight[:49408], model.clip.token_embedding.weight):
                raise ValueError('Original vocabulary changed')
            exposure[category][local] += 1
            update += 1
            for name, value in loss.items():
                totals[name] += float(value.detach()) * len(local)
            if update == 1 or update % 16 == 0 or update % per_epoch == 0:
                report('fit', update, 0 if after_epoch is not None else EPOCHS * per_epoch,
                       category=category, epoch=epoch)
        n = sum(len(counts) for counts in exposure.values())
        histories.append(dict(epoch=epoch, updates=update, **{k: v / n for k, v in totals.items()}))
        write(directory / 'history.json', histories)
        if after_epoch is not None and after_epoch(epoch, update):
            break
    if (update != epoch * per_epoch or any((counts != epoch).any() for counts in exposure.values())
            or frozen_digest(model.clip) != frozen):
        raise ValueError('039 exposure or frozen parameters changed')
    if arm == '039B' and [r['after_update'] for r in refreshes] != list(range(REFRESH_STEPS, update, REFRESH_STEPS)):
        raise ValueError('039B gallery refresh schedule changed')
    path = directory / 'refit_delta.pt'
    atomic_torch_save(dict(state=model.delta_state(), visual=visual.state_dict()), path)
    result = dict(epochs=epoch, updates=update, queries={k: len(v) for k, v in exposure.items()},
                  weight=path.name, weight_sha256=sha(path), history=histories, refreshes=refreshes)
    write(directory / 'fit_results.json', result)
    return result


def evaluate(model, visual, bank, arm, progress, *, query_builder=None, telemetry=None):
    model.eval()
    visual.eval()
    if arm == '039B':
        gallery = score_bank(model.clip.visual, bank, lambda d, t: progress('val_lora_gallery', d, t))
        references = gallery[bank['references']]
    else:
        gallery = bank['gallery']
        ids, inverse = bank['references'].unique(return_inverse=True)
        references = score_bank(model.clip.visual, bank, lambda d, t: progress('val_lora_reference', d, t), ids.cpu().numpy())[inverse]
    scores = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(0, len(references), 8):
            stop = min(start + 8, len(references))
            rows = bank['rows'][start:stop]
            drop, drop_active = bank['drop'][start:stop], bank['drop_active'][start:stop]
            if query_builder is None:
                parts = encode_parts(model, rows, ARM)
                out = compose(model, visual, parts, references[start:stop], drop, drop_active, None, ARM)
            else:
                out = query_builder(model, visual, rows, references[start:stop], drop, drop_active)
            if telemetry is not None:
                telemetry['beta'].extend(out['beta'].cpu().tolist())
                telemetry['valid'].extend(out['valid'].any(1).cpu().tolist())
            with torch.autocast('cuda', enabled=False):
                scores.append((out['q'].float() @ gallery.T).cpu())
            if start % 128 == 0 or stop == len(references):
                progress('official_ranks', stop, len(references))
    result = torch.cat(scores)
    if not torch.isfinite(result).all():
        raise FloatingPointError('039 nonfinite official scores')
    return ranks_for(result, bank)


def run(root, gpu):
    plan = read(root / 'plan.json')
    state = dict(status='running', identity=identity(), gpu=gpu, api_calls=0)
    path = root / 'state.json'
    if path.exists():
        raise FileExistsError(path)
    capacity_mib = 0.
    shared = False

    def progress(stage, done=0, total=0, **fields):
        nonlocal shared
        if (root / 'cancel.request').exists():
            raise InterruptedError('039 cancelled')
        card, other = gpu_snapshot(gpu, plan['gpu_index'])
        reserved = torch.cuda.memory_reserved() / (1 << 20) if capacity_mib else 0.
        projected = card['free_mib'] - max(0., capacity_mib - reserved)
        if projected < 4096:
            state.update(gpu_at_stop=[card, other], projected_free_mib=projected)
            raise CapacityError('039 GPU capacity below safety reserve')
        shared = bool(other)
        state.update(stage=stage, done=done, total=total, projected_free_mib=round(projected, 1), **fields)
        write(path, state)

    try:
        require_space(root, 55)
        if gpu != plan['gpu'] or os.environ.get('CUDA_VISIBLE_DEVICES') != gpu:
            raise ValueError('039 GPU UUID was not bound')
        for name, expected in plan['source_sha256'].items():
            if sha(root / 'code' / name) != expected:
                raise ValueError(f'039 source changed: {name}')
        first = gpu_snapshot(gpu, plan['gpu_index'])
        time.sleep(5)
        second = gpu_snapshot(gpu, plan['gpu_index'])
        state['gpu_startup'] = [first, second]
        if (any(card['free_mib'] < 14336 or card['utilization'] > 75 for card, _ in (first, second))
                or any(pid not in first[1] or used > first[1][pid] + 512 for pid, used in second[1].items())):
            raise RuntimeError('039 GPU lacks stable startup headroom')
        progress('public_clip_load')
        torch.manual_seed(SEED)
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.42)
        capacity_mib = .42 * torch.cuda.get_device_properties(0).total_memory / (1 << 20)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        model, transform = from_public_clip(root / 'weights/open_clip_pytorch_model.bin')

        def pace(_module, _inputs, _output):
            if shared:
                torch.cuda.synchronize()
                time.sleep(.05)

        model.clip.visual.register_forward_hook(pace)
        model.clip.transformer.register_forward_hook(pace)
        cache_root = root / 'prefix_cache'
        cache_root.mkdir()
        prepared, inventory = {}, {}
        for category in CATEGORIES:
            spec = plan['inputs'][category]
            source = root / 'metadata' / f'{category}.json'
            if sha(source) != spec['metadata_sha256']:
                raise ValueError(f'039 training metadata changed: {category}')
            meta = read(source)
            rows = bind_drops(category, 'train', meta['records'], spec['train_archive_sha256'])
            ids = list(dict.fromkeys(meta['banks']['fit'] + meta['banks']['dev']))
            if (len(rows), len(ids)) != (COUNTS[category][0], spec['train_gallery_size']):
                raise ValueError(f'039 training denominator changed: {category}')
            callback = lambda stage, d, t: progress(stage, d, t, category=category)
            target = cache_root / f'{category}_train.npy'
            if 'train_prefix' in spec:
                progress('verify_train_prefix', category=category)
                prefix_source = spec['train_prefix']
                cache = replay_bank(model, transform, category, rows, ids, meta['image_sha256'], prefix_source, callback)
                target = Path(prefix_source['path'])
            else:
                cache = cache_bank(model, transform, category, 'train', rows, ids, meta['image_sha256'], target, callback)
            split_diff = check_split(model.clip.visual, cache)
            bank = full_fit_bank(cache, meta)
            if not torch.equal(cache['gallery'], bank['gallery']):
                raise ValueError('039 prefix rows differ from legal gallery ordering')
            prepared[category] = cache, bank
            inventory[category] = dict(queries=len(rows), gallery=len(ids), prefix_bytes=target.stat().st_size,
                                       frozen_split_max_abs=split_diff)
            write(root / 'train_inventory.json', inventory)
        torch.manual_seed(SEED)
        adapters = list(attach_lora(model.clip.visual))
        initial = model.delta_state()
        initial_diff = check_split(model.clip.visual, prepared['dress'][0])
        write(root / 'lora_preflight.json', dict(rank=8, alpha=16, parameters=sum(p.numel() for a in adapters for p in a.parameters()),
                                                zero_lora_max_abs=initial_diff))
        fitted = {}
        for arm in ('039A', '039B'):
            torch.manual_seed(SEED)
            model.load_delta(initial)
            visual = PixelFreeVisual(1024, True, False).cuda()
            progress('fit_prepare', arm=arm)
            fitted[arm] = fit(root / arm, arm, model, visual, adapters, prepared, progress)
        del prepared, cache, bank
        gc.collect()
        torch.cuda.empty_cache()
        for category in CATEGORIES:
            spec = plan['inputs'][category]
            baseline = Path(spec['control_038g_path'])
            if sha(baseline) != spec['control_038g_sha256']:
                raise ValueError(f'039 official control changed: {category}')
            control = read(baseline)['rows']['038G']['ranks']
            val, hashes, _ = official_bank(category)
            rows = bind_drops(category, 'val', val['rows'], spec['val_archive_sha256'])
            if (len(rows), len(val['ids'])) != COUNTS[category][1:]:
                raise ValueError(f'039 official denominator changed: {category}')
            model.load_delta(initial)
            model.eval()
            callback = lambda stage, d, t: progress(stage, d, t, category=category)
            bank = cache_bank(model, transform, category, 'val', rows, val['ids'], hashes,
                              cache_root / f'{category}_val.npy', callback)
            rows_out = {}
            for arm, fit_result in fitted.items():
                progress('official_prepare', arm=arm, category=category)
                visual = PixelFreeVisual(1024, True, False).cuda()
                checkpoint_path = root / arm / fit_result['weight']
                if sha(checkpoint_path) != fit_result['weight_sha256']:
                    raise ValueError('039 selected checkpoint changed')
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                model.load_delta(checkpoint['state'])
                visual.load_state_dict(checkpoint['visual'], strict=True)
                ranks = evaluate(model, visual, bank, arm, callback)
                metrics = official_result(ranks, control)
                metrics['vs_038G'] = metrics.pop('vs_037')
                if arm == '039B':
                    metrics['vs_039A'] = {str(k): transitions(ranks, rows_out['039A']['ranks'], k) for k in (1, 10, 50)}
                rows_out[arm] = metrics
                write(root / f'{category}_results.json', dict(category=category, queries=len(rows), gallery=len(val['ids']),
                      rows=rows_out, fits=fitted, control_sha256=spec['control_038g_sha256'], api_calls=0))
            del bank
        state.update(status='complete', stage='complete')
        write(path, state)
    except Exception as error:
        state.update(status='failed', error=type(error).__name__, detail=str(error))
        write(path, state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpu', required=True)
    args = parser.parse_args()
    run(args.root, args.gpu)
