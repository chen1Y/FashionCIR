"""Run the 040A/B text ablations on the frozen-gallery 039A protocol."""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch

from experiment_storage import require_space
from fiq_conflict_context_040 import ModificationSlotGate, ResidualTextGate, compose_query, conflict_opr
from fiq_native_caption_math_031 import metrics, transitions
from fiq_pixel_free_038 import PixelFreeVisual
from fiq_vit_lora_039 import attach_lora
from run_029d_freeze_patch import ranks_for
from run_029d_openclip_036 import from_public_clip, gpu_snapshot, official_bank
from run_fiq_native_caption_031 import identity, read, sha, write
from run_fiq_pixel_free_038 import CATEGORIES, COUNTS, SEED, full_fit_bank
from run_fiq_vit_lora_039 import CapacityError, check_split, evaluate, fit, replay_bank
from train_clean_attr_alignment import plain_text
from fiq_conflict_context_040 import ordinary_text


ARMS = ('040A', '040B')


def bind_texts(root, inventory, category, split, rows):
    spec = inventory[f'{category}_{split}']
    path = root / 'texts' / spec['path']
    if sha(path) != spec['sha256']:
        raise ValueError('040 prepared text hash changed')
    for name, digest in spec['source_sha256'].items():
        if sha(Path(name)) != digest:
            raise ValueError('040 reference caption/correction source changed')
    with path.open(encoding='utf-8') as stream:
        texts = [json.loads(line) for line in stream]
    if len(texts) != len(rows) or any(
        any(row[k] != text[k] for k in ('source_index', 'candidate')) for row, text in zip(rows, texts)
    ):
        raise ValueError('040 prepared text/query alignment changed')
    return [dict(row, **text) for row, text in zip(rows, texts)]


def run(root, gpu):
    state_path = root / 'state.json'
    if state_path.exists():
        raise FileExistsError(state_path)
    plan = read(root / 'plan.json')
    state = dict(status='running', identity=identity(), gpu=gpu, api_calls=0)
    capacity = 0.
    shared = False

    def progress(stage, done=0, total=0, **extra):
        nonlocal shared
        if (root / 'cancel.request').exists():
            raise InterruptedError('040 cancelled')
        card, others = gpu_snapshot(gpu, plan['gpu_index'])
        reserved = torch.cuda.memory_reserved() / (1 << 20) if capacity else 0.
        available = card['free_mib'] - max(0., capacity - reserved)
        if available < 4096:
            state.update(gpu_at_stop=[card, others], projected_free_mib=available)
            raise CapacityError('040 GPU capacity below safety reserve')
        shared = bool(others)
        state.update(stage=stage, done=done, total=total, projected_free_mib=round(available, 1), **extra)
        write(state_path, state)

    try:
        require_space(root, 8)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu or plan['gpu'] != gpu:
            raise ValueError('040 GPU UUID mismatch')
        if any(sha(root / 'code' / name) != digest for name, digest in plan['source_sha256'].items()):
            raise ValueError('040 source hash mismatch')
        text_manifest = root / 'texts' / 'inventory.json'
        if sha(text_manifest) != plan['text_inventory_sha256']:
            raise ValueError('040 text inventory changed')
        inventory = read(text_manifest)['splits']
        first = gpu_snapshot(gpu, plan['gpu_index'])
        time.sleep(5)
        second = gpu_snapshot(gpu, plan['gpu_index'])
        state['gpu_startup'] = [first, second]
        if (any(card['free_mib'] < 14336 or card['utilization'] > 75 for card, _ in (first, second))
                or any(pid not in first[1] or used > first[1][pid] + 512 for pid, used in second[1].items())):
            raise CapacityError('040 GPU lacks stable startup headroom')
        progress('public_clip_load')
        torch.manual_seed(SEED)
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.42)
        capacity = .42 * torch.cuda.get_device_properties(0).total_memory / (1 << 20)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        model, transform = from_public_clip(root / 'weights/open_clip_pytorch_model.bin')
        model.attribute_gate = ModificationSlotGate().cuda()
        model.text_mixture = ResidualTextGate().cuda()

        def pace(_module, _inputs, _output):
            if shared:
                torch.cuda.synchronize()
                time.sleep(.05)

        model.clip.visual.register_forward_hook(pace)
        model.clip.transformer.register_forward_hook(pace)
        prepared, features = {}, {}
        for category in CATEGORIES:
            spec = plan['inputs'][category]
            meta_path = root / 'metadata' / f'{category}.json'
            if sha(meta_path) != spec['metadata_sha256']:
                raise ValueError('040 training metadata changed')
            meta = read(meta_path)
            rows = bind_texts(root, inventory, category, 'train', meta['records'])
            ids = list(dict.fromkeys(meta['banks']['fit'] + meta['banks']['dev']))
            if (len(rows), len(ids)) != (COUNTS[category][0], spec['train_gallery_size']):
                raise ValueError('040 training denominator changed')
            if category == CATEGORIES[0]:
                texts = [row['mod_clean'] for row in rows[:8]]
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    diff = float((ordinary_text(model, texts) - plain_text(model, texts)).abs().max())
                if diff > 1e-6:
                    raise ValueError('040 ordinary token encoding differs from 039A')
                write(root / 'text_encoder_preflight.json', dict(samples=len(texts), max_abs=diff))
            progress('verify_train_prefix', category=category)
            callback = lambda stage, d, t: progress(stage, d, t, category=category)
            cache = replay_bank(model, transform, category, rows, ids, meta['image_sha256'], spec['train_prefix'], callback)
            bank = full_fit_bank(cache, meta)
            if not torch.equal(cache['gallery'], bank['gallery']):
                raise ValueError('040 prefix rows differ from legal gallery ordering')
            features[category] = dict(queries=len(rows), gallery=len(ids),
                                      frozen_split_max_abs=check_split(model.clip.visual, cache))
            write(root / 'train_inventory.json', features)
            prepared[category] = cache, bank
        torch.manual_seed(SEED)
        adapters = list(attach_lora(model.clip.visual))
        initial = model.delta_state()
        fits = {}
        for arm in ARMS:
            torch.manual_seed(SEED)
            model.load_delta(initial)
            visual = PixelFreeVisual(1024, True, False).cuda()
            progress('fit_prepare', arm=arm)
            fits[arm] = fit(root / arm, arm, model, visual, adapters, prepared, progress, query_builder=compose_query,
                            auxiliary=conflict_opr if arm == '040B' else None)
        del prepared, cache, bank
        gc.collect()
        torch.cuda.empty_cache()
        scores = {arm: [] for arm in ('039A', *ARMS)}
        for category in CATEGORIES:
            spec = plan['inputs'][category]
            baseline = Path(spec['control_039a_path'])
            if sha(baseline) != spec['control_039a_sha256']:
                raise ValueError('040 official baseline changed')
            control = read(baseline)['rows']['039A']['ranks']
            val, hashes, _ = official_bank(category)
            rows = bind_texts(root, inventory, category, 'val', val['rows'])
            if (len(rows), len(val['ids'])) != COUNTS[category][1:] or len(control) != len(rows):
                raise ValueError('040 official denominator changed')
            model.load_delta(initial)
            model.eval()
            progress('verify_val_prefix', category=category)
            callback = lambda stage, d, t: progress(stage, d, t, category=category)
            bank = replay_bank(model, transform, category, rows, val['ids'], hashes, spec['val_prefix'], callback, split='val')
            out = {'039A': dict(metrics(control), ranks=control)}
            for arm in ARMS:
                progress('official_prepare', arm=arm, category=category)
                visual = PixelFreeVisual(1024, True, False).cuda()
                path = root / arm / fits[arm]['weight']
                if sha(path) != fits[arm]['weight_sha256']:
                    raise ValueError('040 selected weights changed')
                delta = torch.load(path, map_location='cpu', weights_only=True)
                model.load_delta(delta['state'])
                visual.load_state_dict(delta['visual'], strict=True)
                telemetry = {'beta': [], 'valid': []}
                ranks = evaluate(model, visual, bank, arm, callback, query_builder=compose_query, telemetry=telemetry)
                beta = torch.tensor(telemetry['beta'])
                valid = torch.tensor(telemetry['valid'])
                if not torch.isfinite(beta).all() or (beta < 0).any() or (beta > .45).any() or beta[~valid].count_nonzero():
                    raise ValueError('040 text gate invariant changed')
                out[arm] = dict(metrics(ranks), ranks=ranks, beta_mean=float(beta.mean()),
                                beta_max=float(beta.max()), no_valid_slots=int((~valid).sum()),
                                vs_039A={str(k): transitions(ranks, control, k) for k in (1, 10, 50)})
                if arm == '040B':
                    out[arm]['vs_040A'] = {str(k): transitions(ranks, out['040A']['ranks'], k) for k in (1, 10, 50)}
            for arm, result in out.items():
                scores[arm].append(result)
            write(root / f'{category}_results.json',
                  dict(category=category, queries=len(rows), gallery=len(val['ids']), rows=out, fits=fits))
            del bank
        write(root / 'macro_results.json', {arm: {key: sum(item[key] for item in rows) / len(CATEGORIES)
                                                 for key in ('r1', 'r10', 'r50')} for arm, rows in scores.items()})
        state.update(status='complete', stage='complete')
        write(state_path, state)
    except Exception as error:
        state.update(status='failed', error=type(error).__name__, detail=str(error))
        write(state_path, state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpu', required=True)
    args = parser.parse_args()
    run(args.root, args.gpu)
