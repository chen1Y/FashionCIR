"""Refit 039A on GPU 3 with official validation and three-epoch patience."""

import argparse
from fractions import Fraction
import os
from pathlib import Path
import time

import torch

from experiment_storage import atomic_torch_save, require_space
from fiq_native_caption_math_031 import metrics
from fiq_pixel_free_038 import PixelFreeVisual
from fiq_vit_lora_039 import attach_lora
from run_029d_openclip_036 import from_public_clip, gpu_snapshot, official_bank
from run_fiq_native_caption_031 import identity, read, sha, write
from run_fiq_pixel_free_038 import CATEGORIES, COUNTS, SEED, bind_drops, full_fit_bank
from run_fiq_vit_lora_039 import CapacityError, check_split, evaluate, fit, replay_bank


PATIENCE = 3
RECALL_KEYS = ('r1', 'r10', 'r50')


def macro_r10(ranks_by_category):
    recalls = (Fraction(sum(r <= 10 for r in ranks), len(ranks)) for ranks in ranks_by_category.values())
    return sum(recalls) / len(ranks_by_category)


def update_selection(previous, epoch, score):
    if previous is None or score > Fraction(*previous['best_score']):
        return dict(best_epoch=epoch, best_score=[score.numerator, score.denominator], stale=0), True
    return dict(previous, stale=previous['stale'] + 1), False


def run(root, gpu):
    plan = read(root / 'plan.json')
    path = root / 'state.json'
    if path.exists():
        raise FileExistsError(path)
    state = dict(status='running', identity=identity(), gpu=gpu, api_calls=0)
    capacity = 0.
    shared = False

    def progress(stage, done=0, total=0, **extra):
        nonlocal shared
        if (root / 'cancel.request').exists():
            raise InterruptedError('039A early-stop run cancelled')
        card, others = gpu_snapshot(gpu, plan['gpu_index'])
        reserved = torch.cuda.memory_reserved() / (1 << 20) if capacity else 0.
        projected = card['free_mib'] - max(0., capacity - reserved)
        if projected < 4096:
            state.update(gpu_at_stop=[card, others], projected_free_mib=projected)
            raise CapacityError('039A GPU capacity below safety reserve')
        shared = bool(others)
        state.update(stage=stage, done=done, total=total, projected_free_mib=round(projected, 1), **extra)
        write(path, state)

    try:
        require_space(root, 8)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu or plan['gpu'] != gpu or plan['gpu_index'] != 3:
            raise ValueError('039A physical GPU 3 UUID mismatch')
        for name, expected in plan['source_sha256'].items():
            if sha(root / 'code' / name) != expected:
                raise ValueError(f'039A source changed: {name}')
        first = gpu_snapshot(gpu, 3)
        time.sleep(5)
        second = gpu_snapshot(gpu, 3)
        state['gpu_startup'] = [first, second]
        if (any(card['free_mib'] < 14336 or card['utilization'] > 75 for card, _ in (first, second))
                or any(pid not in first[1] or used > first[1][pid] + 512 for pid, used in second[1].items())):
            raise CapacityError('039A GPU 3 lacks stable startup headroom')
        progress('public_clip_load')
        torch.manual_seed(SEED)
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.42)
        capacity = .42 * torch.cuda.get_device_properties(0).total_memory / (1 << 20)
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
        training, validation, inventory = {}, {}, {}
        for category in CATEGORIES:
            spec = plan['inputs'][category]
            meta_file = root / 'metadata' / f'{category}.json'
            if sha(meta_file) != spec['metadata_sha256']:
                raise ValueError(f'039A training metadata changed: {category}')
            meta = read(meta_file)
            rows = bind_drops(category, 'train', meta['records'], spec['train_archive_sha256'])
            ids = list(dict.fromkeys(meta['banks']['fit'] + meta['banks']['dev']))
            if (len(rows), len(ids)) != (COUNTS[category][0], spec['train_gallery_size']):
                raise ValueError(f'039A training denominator changed: {category}')
            progress('verify_train_prefix', category=category)
            report = lambda stage, d, t: progress(stage, d, t, category=category)
            cache = replay_bank(model, transform, category, rows, ids, meta['image_sha256'],
                                spec['train_prefix'], report)
            bank = full_fit_bank(cache, meta)
            if not torch.equal(cache['gallery'], bank['gallery']):
                raise ValueError(f'039A legal gallery order differs: {category}')
            training[category] = cache, bank
            inventory[category] = dict(train_queries=len(rows), train_gallery=len(ids),
                                       train_prefix_max_abs=check_split(model.clip.visual, cache))
            write(root / 'inventory.json', inventory)

        for category in CATEGORIES:
            spec = plan['inputs'][category]
            val, hashes, _ = official_bank(category)
            rows = bind_drops(category, 'val', val['rows'], spec['val_archive_sha256'])
            if (len(rows), len(val['ids'])) != COUNTS[category][1:]:
                raise ValueError(f'039A official validation denominator changed: {category}')
            progress('verify_val_prefix', category=category)
            report = lambda stage, d, t: progress(stage, d, t, category=category)
            bank = replay_bank(model, transform, category, rows, val['ids'], hashes, spec['val_prefix'],
                               report, split='val')
            validation[category] = bank
            inventory[category].update(val_queries=len(rows), val_gallery=len(val['ids']),
                                       val_prefix_max_abs=check_split(model.clip.visual, bank))
            write(root / 'inventory.json', inventory)

        torch.manual_seed(SEED)
        adapters = list(attach_lora(model.clip.visual))
        zero_lora_difference = check_split(model.clip.visual, training['dress'][0])
        write(root / 'lora_preflight.json', dict(rank=8, alpha=16,
              parameters=sum(p.numel() for adapter in adapters for p in adapter.parameters()),
              zero_lora_max_abs=zero_lora_difference))
        torch.manual_seed(SEED)
        visual = PixelFreeVisual(1024, True, False).cuda()
        evaluations = root / 'validation'
        evaluations.mkdir()
        best_path = root / 'best_delta.pt'
        history = []
        selection = None
        best = None

        def after_epoch(epoch, updates):
            nonlocal selection, best
            rows, ranks = {}, {}
            for category, bank in validation.items():
                def report(stage, d, t):
                    progress(stage, d, t, category=category, epoch=epoch, updates=updates)

                progress('official_val_prepare', category=category, epoch=epoch, updates=updates)
                rr = evaluate(model, visual, bank, '039A', report)
                if len(rr) != COUNTS[category][1]:
                    raise ValueError('039A validation did not cover all queries')
                ranks[category] = rr
                rows[category] = dict(metrics(rr), ranks=rr, gallery=len(bank['gallery']))
            macro = {key: sum(v[key] for v in rows.values()) / len(rows) for key in RECALL_KEYS}
            score = macro_r10(ranks)
            selection, improved = update_selection(selection, epoch, score)
            result = dict(epoch=epoch, updates=updates, rows=rows, macro=macro, selection=selection, improved=improved)
            detail = evaluations / f'epoch_{epoch:04d}.json'
            write(detail, result)
            if improved:
                atomic_torch_save(dict(state=model.delta_state(), visual=visual.state_dict(), epoch=epoch), best_path)
                best = dict(epoch=epoch, updates=updates, weight=best_path.name, weight_sha256=sha(best_path),
                            macro=macro, validation=str(detail.relative_to(root)))
                write(root / 'best.json', best)
            history.append(dict(epoch=epoch, updates=updates, macro=macro, **selection, improved=improved,
                                validation=str(detail.relative_to(root)), validation_sha256=sha(detail)))
            write(root / 'validation_history.json', history)
            progress('epoch_validated', epoch, 0, epoch=epoch, updates=updates, stale=selection['stale'],
                     best_epoch=selection['best_epoch'], macro_r10=macro['r10'])
            return selection['stale'] >= PATIENCE

        progress('fit_prepare', arm='039A')
        result = fit(root / '039A', '039A', model, visual, adapters, training, progress, after_epoch=after_epoch)
        if (result['epochs'] != len(history) or selection['stale'] != PATIENCE or best is None
                or sha(best_path) != best['weight_sha256']):
            raise ValueError('039A early-stop or best-checkpoint verification failed')
        write(root / 'summary.json', dict(selection_metric='official_val_equal_category_macro_r10',
                                          patience=PATIENCE, trained_epochs=result['epochs'],
                                          updates=result['updates'], best=best, last=history[-1]))
        state.update(status='complete', stage='complete', best_epoch=selection['best_epoch'])
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
