"""Bind archived Dress/Shoes edits to a fixed 029D transfer split."""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from benchmark_data import BenchmarkData
from fiq_byte_clean_pools import positive_byte_sets
from fiq_full_partition_024 import partition
from fiq_native_caption_math_031 import native_queries
from generate_structured_edits import load_fashioniq_queries
from run_fiq_native_caption_031 import BASE, DATA, read, sha, write


SLOTS = ('color', 'shape', 'detail', 'texture')
SLOT_BY_CATEGORY = {'color': 'color', 'shape': 'shape', 'length': 'shape', 'fit': 'shape',
                    'style': 'shape', 'part': 'detail', 'count': 'detail',
                    'material': 'texture', 'pattern': 'texture'}
LIMITS = {'fit': 1024, 'dev': 256}
OLD = BASE / 'autodl-tmp/results'
SHOES = BASE / 'autodl-tmp/data/Shoes'


def text_path(category, split):
    if category == 'dress':
        return DATA / 'captions' / f'structured_edits_dress_{split}_qwen37flash.json'
    return OLD / 'cirr_shoes_20261015/online_qwen37_v3' / f'structured_shoes_{split}.json'


def slots_for(sample):
    if not (sample['status'] == 'ok' and sample.get('validation', {}).get('valid') is True
            and sample.get('semantic_validation', {}).get('valid') is True):
        return dict.fromkeys(SLOTS, '')
    edit = sample['structured_edit']
    removed = {item['attribute'].strip().casefold() for item in edit['remove']}
    values = {name: [] for name in SLOTS}
    seen = {name: set() for name in SLOTS}
    for action in ('add', 'retain'):
        for item in edit[action]:
            slot = SLOT_BY_CATEGORY.get(item['category'])
            phrase = ' '.join(item['attribute'].split())
            key = phrase.casefold()
            if not slot or not phrase or key in seen[slot] or (action == 'retain' and key in removed):
                continue
            values[slot].append(phrase)
            seen[slot].add(key)
    return {slot: ', '.join(values[slot][:3]) for slot in SLOTS}


def bind_attributes(category, split, rows):
    path = text_path(category, split)
    digest = sha(path)
    samples = read(path)['samples']
    indexed = {sample['source_index']: sample for sample in samples}
    if len(indexed) != len(samples) or set(indexed) != {row['source_index'] for row in rows}:
        raise ValueError(f'{category} {split} edit coverage/identity mismatch')
    result = {}
    for row in rows:
        sample = indexed[row['source_index']]
        if (sample['category'] != category or sample['split'] != split
                or any(row[key] != sample[key] for key in ('candidate', 'target', 'modification_text'))):
            raise ValueError(f"Edit metadata mismatch at {category} {split} {row['source_index']}")
        attrs = slots_for(sample)
        if any(attrs.values()):
            if category == 'dress':
                info = sample.get('generation_input', {})
                safe = (info.get('target_was_provided_to_model') is False
                        and info.get('modification_text') == row['modification_text']
                        and Path(info.get('reference_image', '')).stem == row['candidate'])
            else:
                safe = sample.get('generation_inputs') == 'reference_image_and_modification_only'
            if not safe:
                raise ValueError(f'Unverified generation provenance: {category} {split} {row["source_index"]}')
        result[row['source_index']] = attrs
    if sha(path) != digest:
        raise ValueError(f'Edit source changed while reading: {path}')
    return result, digest


def coverage(rows):
    return dict(queries=len(rows), any_slot=sum(any(row['target_attrs'].values()) for row in rows),
                by_slot={key: sum(bool(row['target_attrs'][key]) for row in rows) for key in SLOTS})


def shoes_partition(dataset):
    hashes = {}
    heldout = {sha(dataset.image_paths[name]) for name in dataset.image_ids['val']}
    for i, name in enumerate(dataset.image_ids['train'], 1):
        hashes[name] = sha(dataset.image_paths[name])
        if i % 2048 == 0:
            print(f'Shoes train JPEG hashes {i}/{len(dataset.image_ids["train"])}', flush=True)
    rows = dataset.queries['train']
    policy = partition(dataset.image_ids['train'], hashes, rows, heldout)
    return dict(all_original_records=rows, image_sha256=hashes,
                known_positive_sha={json.dumps(k): sorted(v) for k, v in positive_byte_sets(rows, hashes).items()},
                records=policy['selected']['fit'] + policy['selected']['dev'],
                banks=policy['banks'], canonical=policy['canonical'], reason_counts=policy['reason_counts'],
                byte_intersections=policy['intersections'])


def prepare(category, path):
    if path.exists():
        raise FileExistsError(path)
    path.mkdir(parents=True)
    if category == 'dress':
        source = BASE / 'runs/fiq-full-024/metadata/dress_train.json'
        prior = read(source)
        if len(prior['records']) != 5639 or any(prior['byte_intersections'].values()):
            raise ValueError('Unexpected Dress isolated fit/dev source')
        keys = ('source_index', 'candidate', 'target', 'modification_text')
        current = load_fashioniq_queries(DATA, 'dress', 'train')
        saved = prior['all_original_records']
        if len(current) != len(saved) or any(any(a[key] != b[key] for key in keys) for a, b in zip(current, saved)):
            raise ValueError('Original Dress metadata changed since 024')
        parent_sha = sha(source)
    else:
        dataset = BenchmarkData(SHOES, 'shoes')
        parent_sha = None
        prior = shoes_partition(dataset)
    all_rows = prior['all_original_records']
    attrs, text_sha = bind_attributes(category, 'train', all_rows)
    selected, offsets, report = [], {}, {}
    for split, limit in LIMITS.items():
        eligible = [row for row in prior['records'] if row['split020'] == split]
        if len(eligible) < limit:
            raise ValueError(f'Insufficient isolated {category} {split} rows: {len(eligible)}')
        def order(row):
            return hashlib.sha256(f'42|029d_transfer|{category}|{row["source_index"]}'.encode()).digest()
        rows = [dict(row, target_attrs=attrs[row['source_index']]) for row in sorted(eligible, key=order)[:limit]]
        offsets[split] = list(range(len(selected), len(selected) + len(rows)))
        selected.extend(rows)
        report[split] = dict(coverage(rows), eligible=len(eligible), gallery=len(prior['banks'][split]))
    payload = {key: prior[key] for key in ('image_sha256', 'known_positive_sha', 'banks', 'canonical',
                                          'byte_intersections', 'reason_counts')}
    payload.update(records=selected, offsets=offsets, source_text_sha256=text_sha, parent_sha256=parent_sha,
                   original_queries=len(all_rows), split_inventory=report)
    write(path / 'train.json', payload)
    print(category, json.dumps(report), flush=True)


def official_inputs(category):
    if category == 'dress':
        rows = load_fashioniq_queries(DATA, 'dress', 'val')
        captions = DATA / 'captions'
        native = native_queries(rows, read(captions / 'cap.dress.val.json'),
                                read(captions / 'image_captions_dress_val.json'),
                                read(captions / 'correction_dict_dress.json'))
        ids = read(DATA / 'image_splits/split.dress.val.json')
        index = OLD / 'fiq_small_improvements_20261026/train_only_candidate_head/official_heldout_byte_index.json.gz'
        hashes = json.loads(gzip.decompress(index.read_bytes()))['image_sha256']
    else:
        data = BenchmarkData(SHOES, 'shoes')
        rows, ids = data.queries['val'], data.image_ids['val']
        native = [{'native_text': data.baseline_text(row, 'val')} for row in rows]
        hashes = None
    attrs, text_sha = bind_attributes(category, 'val', rows)
    records = [dict(row, target_attrs=attrs[row['source_index']], native_text=n['native_text'])
               for row, n in zip(rows, native)]
    return records, ids, hashes, text_sha


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('category', choices=('dress', 'shoes'))
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    prepare(args.category, args.path)
