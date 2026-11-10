"""Evaluate the two native DQU text controls on Dress and Shoes in parallel."""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from fiq_native_caption_math_031 import metrics, transitions
from run_fiq_native_caption_031 import BASE, DATA, identity, read, snapshot, write


CHECKPOINTS = {
    'dress': ('checkpoints/dress_dqu_official_original_seed42_best.pt',
              'a505052a947dfad3b8f1488a66ea316ef3e2688def3c17fae730c80e58cd659f'),
    'shoes': ('results/cirr_shoes_20261015/checkpoints/shoes_dqu_original_seed42_best.pt',
              '0f5972dc791caa78b713dcbac60b83e22c51c7d27d3d6b2de88812985cfaaf2a'),
}
EXPECTED = {'dress': (2017, 3817), 'shoes': (1761, 4658)}
METHODS = ('native_dqu_caption_corrected', 'native_dqu_modification_only')


def load_inputs(category, transform):
    from benchmarkDataset import CIRBenchmark
    from dquDataset import FashionIQ
    from fiq_native_caption_math_031 import native_queries
    from generate_structured_edits import load_fashioniq_queries

    if category == 'shoes':
        dataset = CIRBenchmark(BASE / 'autodl-tmp/data/Shoes', 'shoes', (transform, transform))
        ids = dataset.meta.image_ids['val']
        rows = dataset.meta.queries['val']
        texts = [dataset.meta.baseline_text(row, 'val') for row in rows]

        def written(start, stop):
            return [row['visual_query'] for row in dataset.test_queries[start:stop]]
    else:
        dataset = object.__new__(FashionIQ)
        dataset.category = 'dress'
        dataset.image_dir = DATA / 'resized_image'
        dataset.transform = (transform, transform)
        captions = DATA / 'captions'
        dataset.key_words = read(captions / 'keywords_in_mods_dress.json')
        ids = read(DATA / 'image_splits/split.dress.val.json')
        rows = load_fashioniq_queries(DATA, 'dress', 'val')
        native = native_queries(rows, read(captions / 'cap.dress.val.json'),
                                read(captions / 'image_captions_dress_val.json'),
                                read(captions / 'correction_dict_dress.json'))
        texts = [row['native_text'] for row in native]
        for row in rows:
            if not dataset.key_words.get(row['candidate'] + '_' + row['target']):
                raise ValueError('Missing Dress written-reference keywords')
        for name in ids:
            dataset._image_path(name)

        def written(start, stop):
            return [dataset.get_written_img(row['candidate'], row['target'], 0)[0]
                    for row in rows[start:stop]]

    if (len(rows), len(ids)) != EXPECTED[category] or len(set(ids)) != len(ids):
        raise ValueError(f'Unexpected {category} official validation denominator')
    index = {name: offset for offset, name in enumerate(ids)}
    targets = [index[row['target']] for row in rows]
    references = [index[row['candidate']] for row in rows]
    if sum(a == b for a, b in zip(targets, references)) != (category == 'shoes'):
        raise ValueError('Unexpected reference/target ID collision count')
    return dataset, ids, rows, texts, written, targets, references


def run(root, category):
    import cv2
    import torch
    import torch.nn.functional as F
    from expanded_attr_model import ranks_from_scores
    from fiq_fusion_029 import DQUFusionHead
    from train_attr_slot_pilot import load_clip

    start = time.monotonic()
    state = {'status': 'running', 'dataset': category, 'identity': identity(),
             'api_calls': 0, 'fit_updates': 0, 'phase': 'copy_checkpoint'}
    status_path = root / 'state.json'
    write(status_path, state)

    def progress(phase, completed, total):
        if time.monotonic() - start > 3600:
            raise TimeoutError('One-hour evaluation limit')
        if (root / 'cancel.request').exists():
            raise InterruptedError('Evaluation cancelled')
        state.update(phase=phase, completed=completed, total=total)
        write(status_path, state)

    try:
        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_per_process_memory_fraction(.4)
        relative, expected_sha = CHECKPOINTS[category]
        source = BASE / 'autodl-tmp' / relative
        cache = Path('/dev/shm') / root.parent.name / category
        cache.mkdir(parents=True)
        checkpoint = cache / 'base.pt'
        digest = hashlib.sha256()
        with source.open('rb') as inp, checkpoint.open('xb') as out:
            for block in iter(lambda: inp.read(8 << 20), b''):
                digest.update(block)
                out.write(block)
        if digest.hexdigest() != expected_sha:
            raise ValueError(f'Historical {category} checkpoint SHA mismatch')
        progress('load_checkpoint', 0, 1)
        clip, transform, tokenizer = load_clip(checkpoint)
        head = DQUFusionHead()
        head.from_dqu(checkpoint)
        head.cuda().eval().requires_grad_(False)
        dataset, ids, rows, texts, written, targets, references = load_inputs(category, transform)
        labels = torch.tensor(targets, device='cuda')
        refs = torch.tensor(references, device='cuda')
        ranks = {name: [] for name in METHODS}
        gallery = []
        progress('gallery', 0, len(ids))
        with torch.inference_mode():
            for j in range(0, len(ids), 8):
                images = torch.stack([dataset.get_img(name, 0)[0] for name in ids[j:j+8]]).cuda()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    features = clip.encode_image(images)
                gallery.append(F.normalize(features.float(), dim=-1))
                if j % 128 == 0:
                    progress('gallery', min(j + 8, len(ids)), len(ids))
            gallery = torch.cat(gallery)
            if not torch.isfinite(gallery).all():
                raise FloatingPointError('Nonfinite gallery features')
            progress('queries', 0, len(rows))
            for j in range(0, len(rows), 8):
                stop = min(j + 8, len(rows))
                images = torch.stack(written(j, stop)).cuda()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    visual = F.normalize(clip.encode_image(images).float(), dim=-1)
                    inputs = (texts[j:stop], [row['modification_text'] for row in rows[j:stop]])
                    queries = []
                    for text in inputs:
                        feature = F.normalize(clip.encode_text(tokenizer(text).cuda()).float(), dim=-1)
                        queries.append(head(feature, visual)[0])
                valid = labels[j:stop] != refs[j:stop]
                for name, query in zip(METHODS, queries):
                    scores = query @ gallery.T
                    if not torch.isfinite(scores).all():
                        raise FloatingPointError('Nonfinite retrieval scores')
                    batch_ranks = torch.full((stop - j,), len(ids), device='cuda', dtype=torch.long)
                    batch_ranks[valid] = ranks_from_scores(scores[valid], labels[j:stop][valid], refs[j:stop][valid])
                    ranks[name].extend(batch_ranks.tolist())
                if j % 128 == 0:
                    progress('queries', stop, len(rows))
        result = dict(category=category, queries=len(rows), gallery=len(ids),
                      excluded_reference_targets=sum(a == b for a, b in zip(targets, references)),
                      checkpoint_sha256=expected_sha, api_calls=0, fit_updates=0,
                      precision='BF16 forward, FP32 normalized retrieval scoring',
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / (1 << 30),
                      seconds=time.monotonic() - start,
                      rows={name: dict(metrics(r), ranks=r) for name, r in ranks.items()},
                      caption_minus_modification={str(k): transitions(*ranks.values(), k) for k in (1, 10, 50)})
        write(root / 'results.json', result)
        state.update(status='complete', phase='finished', completed=len(rows), total=len(rows), seconds=result['seconds'])
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write(status_path, state)


def launch(root):
    if root.exists() or (Path('/dev/shm') / root.name).exists():
        raise FileExistsError(root)
    if shutil.disk_usage('/dev/shm').free < 12 * (1 << 30):
        raise RuntimeError('Insufficient space for two checkpoint copies')
    for relative, _ in CHECKPOINTS.values():
        if not (BASE / 'autodl-tmp' / relative).is_file():
            raise FileNotFoundError(relative)
    first = snapshot()
    time.sleep(5)
    second = snapshot()
    selected = [next(d for d in sample['devices'] if d['index'] == 3) for sample in (first, second)]
    uid = selected[0]['uuid']
    if (any(d['uuid'] != uid or d['utilization'] > 10 or d['free_mib'] < 20480 for d in selected)
            or any(line.startswith(uid + ',') for sample in (first, second) for line in sample['compute_apps'])):
        raise RuntimeError(f'GPU 3 did not pass the two-worker resource check: {selected}')
    root.mkdir()
    write(root / 'gpu_selection.json', {'selected_uuid': uid, 'snapshots': [first, second]})
    env = {name: os.environ[name] for name in ('HOME', 'PATH', 'LANG') if name in os.environ}
    env.update(CUDA_VISIBLE_DEVICES=uid, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    workers = {}
    for category in EXPECTED:
        folder = root / category
        folder.mkdir()
        with (folder / 'run.log').open('x') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(folder), '--category', category],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
        workers[category] = process.pid
    write(root / 'launch.json', {'gpu_index': 3, 'gpu_uuid': uid, 'workers': workers,
                                'methods': METHODS, 'api_calls': 0, 'fit_updates': 0})
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
