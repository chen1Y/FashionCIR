"""Train Dress/Shoes 029D on isolated splits and evaluate six frozen controls."""

import argparse
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from attr_transfer_data import SHOES, SLOTS, coverage, official_inputs
from fiq_native_caption_math_031 import COMPARISONS, METHODS, metrics, transitions
from run_dqu_caption_transfer_controls import CHECKPOINTS, EXPECTED
from run_fiq_native_caption_031 import BASE, DATA, identity, read, sha, snapshot, write


ARM = 'full_learned_written'
BASE_CACHE = Path('/dev/shm/dqu-caption-controls-dress-shoes-20261106')


def load_dataset(category, transform):
    from benchmarkDataset import CIRBenchmark
    from dquDataset import FashionIQ

    if category == 'shoes':
        dataset = CIRBenchmark(SHOES, 'shoes', (transform, transform))
        return dataset.meta.image_paths.__getitem__, lambda row, split: dataset._query(row, split)['visual_query']
    dataset = object.__new__(FashionIQ)
    dataset.category = 'dress'
    dataset.image_dir = DATA / 'resized_image'
    dataset.transform = (transform, transform)
    dataset.key_words = read(DATA / 'captions/keywords_in_mods_dress.json')
    return dataset._image_path, lambda row, split: dataset.get_written_img(row['candidate'], row['target'], 0)[0]


def encode_bank(clip, transform, tokenizer, category, split, rows, ids, hashes, progress):
    import torch
    import torch.nn.functional as F
    from PIL import Image

    path_for, write_image = load_dataset(category, transform)
    positions = {name: i for i, name in enumerate(ids)}
    labels = torch.tensor([positions[row['target']] for row in rows], device='cuda')
    references = torch.tensor([positions[row['candidate']] for row in rows], device='cuda')
    checked = {}
    targets = defaultdict(list)
    teacher = phrases = None
    if split == 'train':
        for i, row in enumerate(rows):
            targets[row['target']].append(i)
        text = [row['target_attrs'][slot] or 'unspecified' for row in rows for slot in SLOTS]
        phrases = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for j in range(0, len(text), 32):
                phrases.append(F.normalize(clip.encode_text(tokenizer(text[j:j+32]).cuda()).float(), dim=-1))
        phrases = torch.cat(phrases).reshape(len(rows), 4, -1)
        teacher = torch.full_like(phrases, torch.nan)

    gallery = []
    with torch.no_grad():
        for j in range(0, len(ids), 8):
            names = ids[j:j+8]
            images = []
            for name in names:
                path = path_for(name)
                blob = path.read_bytes()
                digest = hashlib.sha256(blob).hexdigest()
                if hashes is not None and digest != hashes[name]:
                    raise ValueError(f'JPEG bytes changed: {name}')
                checked[name] = digest
                with Image.open(io.BytesIO(blob)) as image:
                    images.append(transform(image.convert('RGB')))
            input_batch = torch.stack(images).cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if any(name in targets for name in names):
                    clip.visual.output_tokens = True
                    features, patches = clip.visual(input_batch)
                    clip.visual.output_tokens = False
                    if clip.visual.final_ln_after_pool:
                        patches = clip.visual.ln_post(patches)
                    patches = F.normalize((patches @ clip.visual.proj).float(), dim=-1).half()
                else:
                    features = clip.encode_image(input_batch)
                    patches = None
            gallery.append(F.normalize(features.float(), dim=-1))
            if patches is not None:
                for k, name in enumerate(names):
                    ix = targets.get(name)
                    if ix:
                        patches_k = patches[k].float()
                        attention = (phrases[ix] @ patches_k.T / .07).softmax(-1)
                        teacher[ix] = F.normalize(attention @ patches_k, dim=-1)
            if j % 128 == 0:
                progress(f'{split}_gallery', j + len(names), len(ids))
        gallery = torch.cat(gallery)
        written = []
        for j in range(0, len(rows), 8):
            batch = rows[j:j+8]
            images = torch.stack([write_image(row, split) for row in batch]).cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                feature = clip.encode_image(images)
            written.append(F.normalize(feature.float(), dim=-1))
            if j % 128 == 0:
                progress(f'{split}_written', j + len(batch), len(rows))
        written = torch.cat(written)
    for name in {row['candidate'] for row in rows}:
        if sha(path_for(name)) != checked[name]:
            raise ValueError(f'Written-reference JPEG changed: {name}')
    if not all(torch.isfinite(x).all() for x in (gallery, written, teacher) if x is not None):
        raise FloatingPointError('Nonfinite image bank or incomplete patch teachers')
    return dict(rows=rows, gallery=gallery, reference=gallery[references], written=written,
                teacher=teacher, positions=positions, labels=labels, references=references,
                image_sha256=checked)


def training_context(bank, meta):
    from expanded_attr_model import legal_masks

    context = {**bank, 'banks': {}}
    for split in ('fit', 'dev'):
        ids = meta['banks'][split]
        ix = meta['offsets'][split]
        positions = {name: i for i, name in enumerate(ids)}
        masks, labels = legal_masks(meta, split)
        references = labels.new_tensor([positions[meta['records'][i]['candidate']] for i in ix])
        context['banks'][split] = dict(indices=ix, gallery=bank['gallery'][[bank['positions'][name] for name in ids]],
                                        labels=labels.cuda(), references=references.cuda(), mask=masks.cuda())
    return context


def fit(root, model, context, progress, *, compose=None, score_dev=None):
    import torch
    from experiment_storage import atomic_torch_save
    from fiq_fusion_029 import encode_parts, fuse_parts, objective
    from train_attr_pair_alignment import frozen_digest
    from train_fiq_fusion_029 import evaluate

    compose = fuse_parts if compose is None else compose
    score_dev = evaluate if score_dev is None else score_dev

    bank = context['banks']['fit']
    n = len(bank['indices'])
    frozen = frozen_digest(model.clip)
    optimizer = torch.optim.AdamW(model.parameter_groups())
    params = [p for group in optimizer.param_groups for p in group['params']]
    rng = torch.Generator().manual_seed(42)
    exposure = torch.zeros(n, dtype=torch.long)
    history = []
    updates = 0

    def backward(local):
        optimizer.zero_grad(set_to_none=True)
        positions = [bank['indices'][j] for j in local.tolist()]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            parts = encode_parts(model, [context['rows'][i] for i in positions], ARM)
            out = compose(model, parts, context['reference'][positions], context['written'][positions], ARM)
            loss = objective(out, bank['gallery'], bank['labels'][local], bank['mask'][local],
                             context['teacher'][positions], ARM)
        if not torch.isfinite(loss['total']):
            raise FloatingPointError('Nonfinite 029D objective')
        loss['total'].backward()
        if model.embedding.weight.grad[:49408].count_nonzero():
            raise ValueError('Old vocabulary received gradients')
        return loss

    def save_epoch(epoch):
        value = score_dev(model, context, 'dev', ARM)
        path = root / f'epoch_{epoch}_delta.pt'
        atomic_torch_save({'state': model.delta_state(), 'epoch': epoch}, path)
        history.append(dict(epoch=epoch, updates=updates, metrics=value, weight=path.name, sha256=sha(path)))
        write(root / 'fit_progress.json', dict(history=history, exposure=exposure.tolist()))

    model.eval()
    loss = backward(torch.arange(8))
    if any(p.grad is not None for p in model.clip.visual.parameters()) or not all(
            p.grad is None or torch.isfinite(p.grad).all() for p in params):
        raise ValueError('Frozen-image or finite-gradient smoke check failed')
    write(root / 'smoke.json', dict(status='passed', batch=8, updates=0,
                                   loss={k: float(v.detach()) for k, v in loss.items()},
                                   peak_gib=torch.cuda.max_memory_allocated() / (1 << 30)))
    optimizer.zero_grad(set_to_none=True)
    save_epoch(0)
    for epoch in range(1, 4):
        model.train()
        order = torch.randperm(n, generator=rng)
        for j in range(0, n, 8):
            local = order[j:j+8]
            loss = backward(local)
            if not torch.isfinite(torch.nn.utils.clip_grad_norm_(params, 1.0)):
                raise FloatingPointError('Nonfinite train gradient norm')
            optimizer.step()
            if not torch.equal(model.embedding.weight[:49408], model.clip.token_embedding.weight):
                raise ValueError('Original vocabulary changed')
            exposure[local] += 1
            updates += 1
            if updates % 16 == 0:
                progress('fit', updates, 3 * (n // 8))
                print(f'epoch={epoch} update={updates} loss={float(loss["total"].detach()):.6f}', flush=True)
        progress('dev', epoch, 3)
        save_epoch(epoch)
    if exposure.tolist() != [3] * n or frozen_digest(model.clip) != frozen:
        raise ValueError('Training exposure or frozen parameters changed')
    selected = max(history, key=lambda h: (h['metrics']['r10'] + h['metrics']['r50'],
                                           -h['metrics']['retrieval_nce'], -h['epoch']))
    write(root / 'fit_results.json', dict(status='complete', history=history, selected=selected,
                                         updates=updates, exposures=exposure.tolist(), frozen_sha256=frozen))
    return selected


def official(root, category, model, bank, base_delta, selected, progress):
    import torch
    import torch.nn.functional as F
    from expanded_attr_model import ranks_from_scores
    from fiq_fusion_029 import encode_parts, fuse_parts
    from train_clean_attr_alignment import plain_text

    rows = bank['rows']
    labels, references = bank['labels'], bank['references']
    gallery, written, reference = bank['gallery'], bank['written'], bank['reference']
    ranks = {name: [] for name in METHODS}
    gates = {name: [] for name in METHODS[2:]}
    native_m = []
    model.load_delta(base_delta)
    model.eval()
    native_head = copy.deepcopy(model.image_fusion).eval()

    def rank(query, start):
        stop = start + len(query)
        scores = query.float() @ gallery.T
        if not torch.isfinite(scores).all():
            raise FloatingPointError('Nonfinite official scores')
        keep = labels[start:stop] != references[start:stop]
        result = torch.full((len(query),), len(gallery), dtype=torch.long, device='cuda')
        result[keep] = ranks_from_scores(scores[keep], labels[start:stop][keep], references[start:stop][keep])
        return result.tolist()

    with torch.inference_mode():
        for j in range(0, len(rows), 8):
            batch = rows[j:j+8]
            w = written[j:j+len(batch)]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                m = F.normalize(model.clip.encode_text(model.tokenizer([r['native_text'] for r in batch]).cuda()).float(), dim=-1)
                raw = F.normalize(model.clip.encode_text(model.tokenizer([r['modification_text'] for r in batch]).cuda()).float(), dim=-1)
                q, _ = native_head(m, w)
                qraw, _ = native_head(raw, w)
            native_m.append(m)
            ranks[METHODS[0]].extend(rank(q, j))
            ranks[METHODS[1]].extend(rank(qraw, j))
            if j % 128 == 0:
                progress('official_baselines', j + len(batch), len(rows))
    weights = root / selected['weight']
    if sha(weights) != selected['sha256']:
        raise ValueError('Selected checkpoint SHA changed')
    model.load_delta(torch.load(weights, map_location='cpu', weights_only=True)['state'])
    proxy = SimpleNamespace(attribute_gate=model.attribute_gate, text_mixture=model.text_mixture,
                            image_fusion=native_head)
    max_anchor_diff = 0.0
    with torch.inference_mode():
        for j in range(0, len(rows), 8):
            batch = rows[j:j+8]
            w, v = written[j:j+len(batch)], reference[j:j+len(batch)]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                parts = encode_parts(model, batch, ARM)
                mcap = plain_text(model, [row['native_text'] for row in batch])
                raw = fuse_parts(model, parts, v, w, ARM)
                native = fuse_parts(model, dict(parts, mod=mcap), v, w, ARM)
                q0, alpha0 = model.image_fusion(mcap, w)
                m = native_m[j // 8]
                anchor = fuse_parts(proxy, dict(parts, mod=m), v, w, ARM)
                qanchor, _ = native_head(F.normalize(m, dim=-1), w)
                qnative, _ = native_head(m, w)
            max_anchor_diff = max(max_anchor_diff, float((qnative - qanchor).abs().max()))
            if max_anchor_diff > 1e-6 or rank(qanchor, j) != ranks[METHODS[0]][j:j+len(batch)]:
                raise ValueError('Native anchor beta=0 identity check failed')
            beta0 = dict(q=q0, alpha=alpha0, beta=torch.zeros_like(alpha0))
            for name, result in zip(METHODS[2:], (raw, beta0, native, anchor)):
                ranks[name].extend(rank(result['q'], j))
                gates[name].extend(zip(result['alpha'].tolist(), result['beta'].tolist()))
            if j % 128 == 0:
                progress('official_029d_controls', j + len(batch), len(rows))
    result = dict(category=category, queries=len(rows), gallery=len(gallery), selected=selected['epoch'],
                  native_anchor_beta0_max_abs=max_anchor_diff, attribute_coverage=coverage(rows),
                  reference_target_collisions=int((labels == references).sum()), api_calls=0,
                  rows={k: dict(metrics(v), ranks=v) for k, v in ranks.items()}, gates=gates,
                  comparisons={a+'_minus_'+b: {str(k): transitions(ranks[a], ranks[b], k) for k in (1,10,50)}
                               for a,b in COMPARISONS})
    write(root / 'official_results.json', result)
    print(json.dumps({name: metrics(value) for name, value in ranks.items()}), flush=True)


def run(root, category):
    start = time.monotonic()
    state = dict(status='running', phase='loading', category=category, identity=identity(), api_calls=0)
    path = root / 'state.json'
    if path.exists():
        raise FileExistsError(path)
    write(path, state)

    def progress(phase, completed, total):
        if time.monotonic() - start > 3600:
            raise TimeoutError('One-hour category limit')
        if (root.parent / 'cancel.request').exists():
            raise InterruptedError('Run cancelled')
        state.update(phase=phase, completed=completed, total=total)
        write(path, state)

    try:
        import cv2
        import torch
        from fiq_fusion_029 import FusionText029
        from train_attr_slot_pilot import load_clip

        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_per_process_memory_fraction(.4)
        checkpoint = BASE_CACHE / category / 'base.pt'
        if sha(checkpoint) != CHECKPOINTS[category][1]:
            raise ValueError('Cached DQU checkpoint SHA mismatch')
        clip, transform, tokenizer = load_clip(checkpoint)
        meta = read(root / 'train.json')
        bank = encode_bank(clip, transform, tokenizer, category, 'train', meta['records'],
                           meta['banks']['fit'] + meta['banks']['dev'], meta['image_sha256'], progress)
        progress('legal_masks', 0, 1)
        context = training_context(bank, meta)
        model = FusionText029(clip, tokenizer, 2).cuda()
        model.image_fusion.from_dqu(checkpoint)
        model.configure_arm(ARM)
        base_delta = model.delta_state()
        progress('fit_smoke', 0, 1)
        selected = fit(root, model, context, progress)
        del context, bank
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        rows, ids, hashes, text_sha = official_inputs(category)
        if (len(rows), len(ids)) != EXPECTED[category]:
            raise ValueError('Official denominator changed')
        write(root / 'official_inputs.json', dict(queries=len(rows), gallery=len(ids), source_text_sha256=text_sha,
                                                  attribute_coverage=coverage(rows)))
        bank = encode_bank(clip, transform, tokenizer, category, 'val', rows, ids, hashes, progress)
        official(root, category, model, bank, base_delta, selected, progress)
        state.update(status='complete', phase='finished', selected_epoch=selected['epoch'],
                     peak_allocated_gib=torch.cuda.max_memory_allocated() / (1 << 30), seconds=time.monotonic()-start)
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write(path, state)


def launch(root):
    if (root / 'launch.json').exists():
        raise FileExistsError('Existing launch')
    for category in EXPECTED:
        folder = root / category
        if not (folder / 'train.json').is_file() or (folder / 'state.json').exists():
            raise RuntimeError(f'{category} preflight/worker state does not permit launch')
        if not (BASE_CACHE / category / 'base.pt').is_file():
            raise FileNotFoundError(BASE_CACHE / category / 'base.pt')
    first = snapshot()
    time.sleep(5)
    second = snapshot()
    cards = [next(d for d in sample['devices'] if d['index'] == 3) for sample in (first, second)]
    uid = cards[0]['uuid']
    if (any(d['uuid'] != uid or d['utilization'] > 10 or d['free_mib'] < 20480 for d in cards)
            or any(line.startswith(uid + ',') for sample in (first, second) for line in sample['compute_apps'])):
        raise RuntimeError('GPU 3 failed the two-process resource check')
    env = {name: os.environ[name] for name in ('HOME', 'PATH', 'LANG') if name in os.environ}
    env.update(CUDA_VISIBLE_DEVICES=uid, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    workers = {}
    for category in EXPECTED:
        folder = root / category
        with (folder / 'run.log').open('x') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(folder), '--category', category],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
        workers[category] = process.pid
    write(root / 'launch.json', dict(workers=workers, gpu=cards[1], snapshots=[first, second], api_calls=0,
                                    source_sha256=sha(__file__), inputs={c:sha(root/c/'train.json') for c in EXPECTED}))
    print(workers, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--category', choices=EXPECTED)
    args = parser.parse_args()
    if args.category:
        run(args.root, args.category)
    else:
        launch(args.root)
