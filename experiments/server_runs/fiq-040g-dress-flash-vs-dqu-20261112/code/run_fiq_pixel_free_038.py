"""Matched full-train 038 visual ablations with fixed 037 training budgets."""

import argparse
import gc
import math
import os
import time
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from attr_slot_data import slot_view
from attr_transfer_data import text_path
from experiment_storage import atomic_torch_save, require_space
from fiq_fusion_029 import encode_parts
from fiq_native_caption_math_031 import metrics, transitions
from fiq_pixel_free_038 import PixelFreeVisual, compose, loss_for
from run_029d_freeze_patch import patch_scores, ranks_for
from run_029d_full_train_037 import BATCH, RATES, SEED, full_fit_bank
from run_029d_openclip_036 import OPENCLIP_SHA, from_public_clip, gpu_snapshot, official_bank
from run_attr_transfer_029d import ARM, encode_bank
from run_fiq_native_caption_031 import BASE, identity, read, sha, write
from train_attr_pair_alignment import frozen_digest


CATEGORIES = ('dress', 'shirt', 'toptee')
PARENT = BASE / 'runs/fiq-029d-full-train-037-gpu2-restart01'
COUNTS = {'dress': (5639, 2017, 3817), 'shirt': (5312, 2038, 6346), 'toptee': (5291, 1961, 5373)}
ABLATIONS = {
    '038B_clean': ('pure_ref', False, False),
    '038C_erase': ('pure_ref', True, False),
    '038D_proxy': ('pure_ref', True, True),
}


def archive_path(category, split):
    if category == 'dress':
        return text_path(category, split)
    return (BASE / 'autodl-tmp/results/fiq_shirt_toptee_20261022' / category
            / f'structured_{category}_{split}.json')


def bind_drops(category, split, rows, source_sha):
    path = archive_path(category, split)
    if sha(path) != source_sha:
        raise ValueError(f'038 archived {category} {split} edits changed')
    samples = read(path)['samples']
    by_index = {sample['source_index']: sample for sample in samples}
    if len(by_index) != len(samples):
        raise ValueError('Duplicate source indices in structured archive')
    bound = []
    for row in rows:
        sample = by_index[row['source_index']]
        if (sample['category'] != category or sample['split'] != split
                or any(sample[k] != row[k] for k in ('candidate', 'target', 'modification_text'))):
            raise ValueError(f"038 archive row mismatch: {category} {split} {row['source_index']}")
        text = slot_view(sample, row['modification_text'])['drop_attrs']
        if text:
            provenance = sample['generation_input']
            if (provenance['target_was_provided_to_model'] is not False
                    or provenance['modification_text'] != row['modification_text']
                    or Path(provenance['reference_image']).stem != row['candidate']):
                raise ValueError('Unsafe drop-attribute provenance')
        bound.append(dict(row, drop_attrs=text))
    return bound


def drop_features(model, rows):
    phrases = list(dict.fromkeys(row['drop_attrs'] for row in rows if row['drop_attrs']))
    positions = {text: i + 1 for i, text in enumerate(phrases)}
    dimension = model.clip.text_projection.shape[-1]
    features = [torch.zeros((1, dimension), device='cuda')]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for offset in range(0, len(phrases), 32):
            tokens = model.tokenizer(phrases[offset:offset + 32]).cuda()
            features.append(F.normalize(model.clip.encode_text(tokens).float(), dim=-1))
    ids = torch.tensor([positions.get(row['drop_attrs'], 0) for row in rows], device='cuda')
    result = torch.cat(features)[ids]
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite conflict text feature')
    return result, ids != 0


def make_bank(model, transform, category, split, rows, ids, hashes, include_written, progress):
    bank = encode_bank(model.clip, transform, model.tokenizer, category, split, rows, ids, hashes,
                       progress, include_written=include_written)
    bank['drop'], bank['drop_active'] = drop_features(model, rows)
    return bank


def forward(model, visual, cache, positions):
    parts = encode_parts(model, [cache['rows'][i] for i in positions], ARM)
    written = cache['written'][positions] if visual is None else None
    return compose(model, visual, parts, cache['reference'][positions], cache['drop'][positions],
                   cache['drop_active'][positions], written, ARM)


def fit(root, model, visual, categories, training, rate, epochs, a, progress, *, dress_loss_weight=1.):
    groups = model.parameter_groups()
    if visual is not None and list(visual.parameters()):
        groups.append(dict(params=list(visual.parameters()), lr=1e-4, weight_decay=.01))
    for group in groups:
        group['lr'] *= rate
    params = [p for group in groups for p in group['params']]
    optimizer = torch.optim.AdamW(groups)
    frozen = frozen_digest(model.clip)
    shuffle = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)
    exposure = {cat: torch.zeros(len(training[cat][1]['indices']), dtype=torch.int64) for cat in categories}
    per_epoch = sum(math.ceil(len(counts) / BATCH) for counts in exposure.values())
    update = 0
    histories = []
    for epoch in range(1, epochs + 1):
        model.train()
        if visual is not None:
            visual.train()
        batches = [(cat, ix) for cat in categories
                   for ix in torch.randperm(len(exposure[cat]), generator=shuffle).split(BATCH)]
        if len(categories) > 1:
            batches = [batches[i] for i in torch.randperm(len(batches), generator=shuffle).tolist()]
        totals = Counter()
        for cat, local in batches:
            cache, bank = training[cat]
            positions = [bank['indices'][i] for i in local.tolist()]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = forward(model, visual, cache, positions)
                loss = loss_for(out, bank['gallery'], bank['labels'][local], bank['mask'][local],
                                cache['teacher'][positions], cache['reference'][positions], ARM,
                                lambda_rec=a.lambda_rec, lambda_kl=a.lambda_kl,
                                reference_weight=.2 if visual is not None and visual.proxy is not None else 0.,
                                lambda_gallery_kl=.5 if a.variant in ('038E', '038G') else 0.)
            if not all(torch.isfinite(value) for value in loss.values()):
                raise FloatingPointError('Nonfinite 038 objective')
            if a.variant == '038G':
                category_weight = dress_loss_weight if cat == 'dress' else 1.
                loss['optimization_loss'] = loss['total'] * category_weight
                loss['optimization_loss'].backward()
            else:
                loss['total'].backward()
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            if not torch.equal(model.embedding.weight[:49408], model.clip.token_embedding.weight):
                raise ValueError('Original vocabulary changed')
            exposure[cat][local] += 1
            update += 1
            for name, value in loss.items():
                totals[name] += float(value.detach()) * len(local)
            if update % 16 == 0 or update % per_epoch == 0:
                progress('fit', update, epochs * per_epoch, epoch=epoch)
        n = sum(len(counts) for counts in exposure.values())
        histories.append(dict(epoch=epoch, updates=update, **{k: v / n for k, v in totals.items()}))
        write(root / 'history.json', histories)
    if (update != epochs * per_epoch or any((counts != epochs).any() for counts in exposure.values())
            or frozen_digest(model.clip) != frozen):
        raise ValueError('038 training exposure or frozen parameters changed')
    path = root / 'refit_delta.pt'
    atomic_torch_save(dict(state=model.delta_state(), visual=visual.state_dict() if visual is not None else None), path)
    result = dict(epochs=epochs, rate_multiplier=rate, updates=update, queries={k: len(v) for k, v in exposure.items()},
                  weight=path.name, weight_sha256=sha(path), history=histories)
    if a.variant == '038G':
        result.update(dress_loss_weight=dress_loss_weight, lambda_gallery_kl=.5, tau_gallery_kl=.07)
    write(root / 'fit_results.json', result)
    return result


def evaluate(model, visual, bank, progress):
    model.eval()
    if visual is not None:
        visual.eval()
    scores, slots, valid, alpha = [], [], [], []
    n = len(bank['rows'])
    with torch.inference_mode():
        for start in range(0, n, 8):
            stop = min(start + 8, n)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = forward(model, visual, bank, list(range(start, stop)))
            scores.append((out['q'].float() @ bank['gallery'].T).cpu())
            slots.append(out['slots'].float())
            valid.append(out['valid'])
            alpha.append(out['alpha'].float())
            if start % 128 == 0 or stop == n:
                progress('official_ranks', stop, n)
    values = torch.cat(scores)
    if not torch.isfinite(values).all():
        raise FloatingPointError('Nonfinite 038 official score')
    return values, torch.cat(slots), torch.cat(valid), float(torch.cat(alpha).mean())


def official_result(ranks, control):
    return dict(**metrics(ranks), ranks=ranks,
                vs_037={str(k): transitions(ranks, control, k) for k in (1, 10, 50)})


def completed_fits(resume, arms, category, rate, epochs):
    source = Path(resume['directory'])
    inventory = source / 'train_inventory.json'
    if sha(inventory) != resume['inventory_sha256']:
        raise ValueError('038 completed training inventory changed')
    fitted, weights = {}, {}
    for name in arms:
        arm_root = source / name
        report = arm_root / 'fit_results.json'
        if sha(report) != resume['fit_sha256'][name]:
            raise ValueError(f'038 completed fit report changed: {name}')
        fitted[name] = result = read(report)
        n = COUNTS[category][0]
        if ((result['epochs'], result['rate_multiplier'], result['queries'], result['updates'])
                != (epochs, rate, {category: n}, epochs * math.ceil(n / BATCH))):
            raise ValueError(f'038 completed fit budget mismatch: {name}')
        weights[name] = arm_root / result['weight']
        if sha(weights[name]) != result['weight_sha256']:
            raise ValueError(f'038 completed weight changed: {name}')
    return read(inventory), fitted, weights


def run(a):
    root = a.root
    plan = read(root / 'plan.json')
    categories = tuple(a.joint_categories.split(',')) if a.joint_categories else (a.category,)
    arms = (ABLATIONS if a.suite else {'custom': (a.visual_input_mode, a.use_conflict_eraser, a.use_latent_proxy)})
    if a.variant:
        if (a.suite or a.use_latent_proxy or a.visual_input_mode != 'pure_ref'
                or (a.variant == '038E' and a.joint_categories)
                or (a.variant in ('038F', '038G') and (not a.joint_categories or categories != CATEGORIES))):
            raise ValueError('038E requires a category and 038F/038G require the registered joint split')
        arms = {a.variant: ('pure_ref', True, False)}
        if a.variant == '038G':
            arms['038G_dress_weight_1p2'] = ('pure_ref', True, False)
    if (not categories or len(set(categories)) != len(categories) or any(cat not in CATEGORIES for cat in categories)
            or any(mode == 'written_legacy' and (erase or proxy) for mode, erase, proxy in arms.values())
            or any(not math.isfinite(v) or v < 0 for v in (a.lambda_rec, a.lambda_kl, a.local_patch_weight))):
        raise ValueError('Invalid 038 configuration')
    if a.suite and (a.lambda_rec, a.lambda_kl, a.local_patch_weight) != (.2, .5, .15):
        raise ValueError('Registered 038 suite requires the fixed loss and reranking weights')
    group = root / a.variant if a.variant else root
    directory = group / ('joint_' + '_'.join(categories) if a.joint_categories else a.category)
    directory.mkdir()
    state_path = directory / 'state.json'
    state = dict(status='running', identity=identity(), gpu=a.gpu, categories=categories,
                 exclusive_gpu=a.exclusive_gpu, variant=a.variant, api_calls=0)

    capacity_mib = 0.
    shared_gpu = False

    def progress(stage, done=0, total=0, **extra):
        nonlocal shared_gpu
        if (root / 'cancel.request').exists():
            raise InterruptedError('038 cancelled')
        card, other = gpu_snapshot(a.gpu, plan['gpu_index'])
        reserved_mib = torch.cuda.memory_reserved() / (1 << 20) if capacity_mib else 0.
        potential_growth_mib = max(0., capacity_mib - reserved_mib)
        headroom_mib = card['free_mib'] - potential_growth_mib
        if headroom_mib < 4096 or (a.exclusive_gpu and other):
            state['gpu_at_stop'] = [card, other]
            state['projected_free_mib'] = round(headroom_mib, 1)
            raise RuntimeError('038 GPU capacity below safety reserve')
        shared_gpu = bool(other)
        state.update(stage=stage, done=done, total=total, forward_pause_seconds=.05 if shared_gpu else 0.,
                     gpu_free_mib=card['free_mib'], projected_free_mib=round(headroom_mib, 1), **extra)
        write(state_path, state)

    try:
        require_space(root, 20)
        if a.gpu != plan['gpu'] or os.environ.get('CUDA_VISIBLE_DEVICES') != a.gpu:
            raise ValueError('038 GPU UUID was not bound')
        for name, digest in plan['source_sha256'].items():
            if sha(root / 'code' / name) != digest:
                raise ValueError(f'038 source changed: {name}')
        first = gpu_snapshot(a.gpu, plan['gpu_index'])
        time.sleep(5)
        second = gpu_snapshot(a.gpu, plan['gpu_index'])
        state['gpu_startup'] = [first, second]
        if (any(g['free_mib'] < 14336 or g['utilization'] > 75 for g, _ in (first, second))
                or any(pid not in first[1] or used > first[1][pid] + 512 for pid, used in second[1].items())):
            raise RuntimeError('038 GPU lacks stable shared headroom')
        if a.exclusive_gpu and (first[1] or second[1]):
            raise RuntimeError('038 exclusive GPU has another compute process')
        allowed = {pid: max(used, first[1][pid]) for pid, used in second[1].items()}
        state['forward_pause_seconds'] = .05 if allowed else 0.
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
            if shared_gpu:
                torch.cuda.synchronize()
                time.sleep(.05)
        model.clip.visual.register_forward_hook(pace)
        model.clip.transformer.register_forward_hook(pace)

        include_written = any(mode == 'written_legacy' for mode, _, _ in arms.values())
        training = {}
        train_inventory = {}
        if a.joint_categories:
            rate, epochs = 1., 3
        else:
            spec = plan['inputs'][a.category]
            rate, epochs = RATES[spec['profile']], spec['epochs']
        initial = model.delta_state()
        resume = plan.get('resume_fit', {}).get(a.category)
        if resume:
            if a.joint_categories or (not a.suite and a.variant != '038E'):
                raise ValueError('038 evaluation resume requires the registered per-category experiment')
            train_inventory, fitted, weights = completed_fits(resume, arms, a.category, rate, epochs)
            write(directory / 'train_inventory.json', train_inventory)
            progress('completed_fit_reused', reused_fits=len(fitted))
        else:
            for cat in categories:
                source = root / 'metadata' / f'{cat}.json'
                spec = plan['inputs'][cat]
                if sha(source) != spec['metadata_sha256']:
                    raise ValueError(f'038 prepared {cat} metadata changed')
                meta = read(source)
                rows = bind_drops(cat, 'train', meta['records'], spec['train_archive_sha256'])
                if len(rows) != COUNTS[cat][0]:
                    raise ValueError('038 expected every eligible 037 train query')
                ids = list(dict.fromkeys(meta['banks']['fit'] + meta['banks']['dev']))
                if a.variant == '038G' and len(ids) != spec['train_gallery_size']:
                    raise ValueError(f'038G train gallery denominator changed: {cat}')
                callback = lambda stage, done, total: progress(stage, done, total, category=cat)
                cache = make_bank(model, transform, cat, 'train', rows, ids, meta['image_sha256'], include_written, callback)
                training[cat] = cache, full_fit_bank(cache, meta)
                train_inventory[cat] = dict(queries=len(rows), gallery=len(ids), drop_active=int(cache['drop_active'].sum()))
            write(directory / 'train_inventory.json', train_inventory)
            fitted, weights = {}, {}
            for name, (mode, erase, proxy) in arms.items():
                arm_root = directory / name
                arm_root.mkdir()
                torch.manual_seed(SEED)
                model.load_delta(initial)
                visual = None if mode == 'written_legacy' else PixelFreeVisual(1024, erase, proxy).cuda()
                progress('fit_prepare', arm=name)
                dress_loss_weight = 1.2 if name == '038G_dress_weight_1p2' else 1.
                fitted[name] = fit(arm_root, model, visual, categories, training, rate, epochs, a, progress,
                                   dress_loss_weight=dress_loss_weight)
                weights[name] = arm_root / fitted[name]['weight']
            del cache
        del training
        gc.collect()
        torch.cuda.empty_cache()

        for cat in categories:
            spec = plan['inputs'][cat]
            parent_result = PARENT / cat / 'results.json'
            if sha(parent_result) != spec['control_sha256']:
                raise ValueError('037 official control changed')
            control = read(parent_result)['rows']['raw']['ranks']
            control_c = None
            if a.variant:
                source_c = Path(plan['control_038c_directory']) / cat / f'{cat}_results.json'
                if sha(source_c) != spec['control_038c_sha256']:
                    raise ValueError('038C official control changed')
                control_c = read(source_c)['rows']['038C_erase']['ranks']
            joint_controls = {}
            if a.variant == '038G':
                for variant, reference in spec['controls_038g'].items():
                    source = Path(reference['path'])
                    if sha(source) != reference['sha256']:
                        raise ValueError(f'Official control changed: {variant}')
                    joint_controls[variant] = read(source)['rows'][variant]['ranks']
            val, hashes, _ = official_bank(cat)
            rows = bind_drops(cat, 'val', val['rows'], spec['val_archive_sha256'])
            if (len(rows), len(val['ids'])) != COUNTS[cat][1:]:
                raise ValueError('038 official denominator changed')
            callback = lambda stage, done, total: progress(stage, done, total, category=cat)
            model.load_delta(initial)
            model.eval()
            bank = make_bank(model, transform, cat, 'val', rows, val['ids'], hashes, include_written, callback)
            output = {}
            for name, (mode, erase, proxy) in arms.items():
                progress('official_prepare', arm=name, category=cat)
                visual = None if mode == 'written_legacy' else PixelFreeVisual(1024, erase, proxy).cuda()
                path = weights[name]
                if sha(path) != fitted[name]['weight_sha256']:
                    raise ValueError('038 selected checkpoint changed')
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                model.load_delta(checkpoint['state'])
                if visual is not None:
                    visual.load_state_dict(checkpoint['visual'], strict=True)
                scores, slots, valid, mean_alpha = evaluate(model, visual, bank, progress)
                ranks = ranks_for(scores, bank)
                output[name] = dict(**official_result(ranks, control), mean_text_alpha=mean_alpha)
                if control_c is not None:
                    output[name]['vs_038C'] = {str(k): transitions(ranks, control_c, k) for k in (1, 10, 50)}
                for variant, reference_ranks in joint_controls.items():
                    output[name][f'vs_{variant}'] = {str(k): transitions(ranks, reference_ranks, k) for k in (1, 10, 50)}
                if name == '038G_dress_weight_1p2':
                    output[name]['vs_038G'] = {str(k): transitions(ranks, output['038G']['ranks'], k) for k in (1, 10, 50)}
                if erase:
                    output[name]['gamma_erase'] = float(visual.eraser.gamma.detach().clamp(0, 1.5))
                if proxy:
                    output[name]['alpha_proxy'] = float(.75 * visual.proxy.blend_logit.detach().sigmoid())
                if proxy and a.local_patch_weight:
                    local, patch_audit = patch_scores(model.clip, transform, cat, bank, slots, valid,
                                                      lambda d, t: progress('official_patch', d, t))
                    patch_ranks = ranks_for(scores + a.local_patch_weight * local, bank)
                    output[name + '_patch'] = dict(**official_result(patch_ranks, control), patch_audit=patch_audit)
                write(directory / f'{cat}_results.json', dict(category=cat, queries=len(rows), gallery=len(val['ids']),
                      train=train_inventory[cat], val_drop_active=int(bank['drop_active'].sum()), rows=output,
                      fits=fitted, control_sha256=spec['control_sha256'], openclip_sha256=OPENCLIP_SHA, api_calls=0))
            del bank
        state.update(status='complete', stage='complete')
        write(state_path, state)
    except Exception as error:
        state.update(status='failed', error=type(error).__name__, detail=str(error))
        write(state_path, state)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--gpu', required=True)
    p.add_argument('--exclusive_gpu', action='store_true', help='Require no other GPU compute process at startup')
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument('--category', choices=CATEGORIES)
    target.add_argument('--joint_categories')
    p.add_argument('--suite', action='store_true')
    p.add_argument('--variant', choices=('038E', '038F', '038G'))
    p.add_argument('--visual_input_mode', choices=('pure_ref', 'written_legacy'), default='pure_ref')
    p.add_argument('--use_conflict_eraser', action='store_true')
    p.add_argument('--use_latent_proxy', action='store_true')
    p.add_argument('--lambda_rec', type=float, default=.2)
    p.add_argument('--lambda_kl', type=float, default=.5)
    p.add_argument('--local_patch_weight', type=float, default=.15)
    run(p.parse_args())
