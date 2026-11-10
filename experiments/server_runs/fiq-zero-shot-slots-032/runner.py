"""Text-query zero-shot retrieval with raw OpenCLIP and fixed attribute pooling."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import traceback

from attr_slot_data import SLOTS
from fiq_native_caption_math_031 import metrics, transitions
from run_fiq_native_caption_031 import DATA, identity, read, sha, write


MODEL_SHA = '0084e75319a50ad85ef45377bad5bc38f2f58824459eb690048d51c9f8863be55'
EXPECTED = {'dress': (2017, 3817), 'shirt': (2038, 6346), 'toptee': (1961, 5373),
            'shoes': (1761, 4658)}
METHODS = ('raw_mod', 'align_slots', 'align_slots_eos', 'native_eos_slots')


def prepare(category, pilot):
    if category in ('shirt', 'toptee'):
        from run_029d_gate_refresh import inputs_029

        metadata, evidence = inputs_029(category, 'val')
        rows, ids, hashes = (metadata[k] for k in ('records', 'gallery_ids', 'image_sha256'))
        text_sha = evidence['text_sha256']
    else:
        from attr_transfer_data import official_inputs

        rows, ids, hashes, text_sha = official_inputs(category)
    if (len(rows), len(ids)) != EXPECTED[category] or len(set(ids)) != len(ids):
        raise ValueError('Unexpected official query or gallery count')
    if pilot:
        def key(kind, value):
            return hashlib.sha256(f'032|{category}|{kind}|{value}'.encode()).digest()

        selected = sorted(rows, key=lambda row: key('query', row['source_index']))[:64]
        selected_indices = {row['source_index'] for row in selected}
        rows = [row for row in rows if row['source_index'] in selected_indices]
        required = {row[k] for row in rows for k in ('candidate', 'target')}
        distractors = sorted(set(ids) - required, key=lambda name: key('image', name))[:256]
        included = required.union(distractors)
        ids = [name for name in ids if name in included]
    if any(not isinstance(row['modification_text'], str) or not row['modification_text'].strip()
           or set(row['target_attrs']) != set(SLOTS)
           or any(not isinstance(x, str) for x in row['target_attrs'].values()) for row in rows):
        raise ValueError('Invalid modification or attribute schema')
    return rows, ids, hashes, text_sha


def image_paths(category):
    if category == 'shoes':
        from attr_transfer_data import SHOES
        from benchmark_data import BenchmarkData

        return BenchmarkData(SHOES, 'shoes').image_paths.__getitem__

    def fashioniq_path(name):
        root = DATA / 'resized_image'
        flat = root / f'{name}.jpg'
        return flat if flat.is_file() else root / category / f'{name}.jpg'

    return fashioniq_path


def run_category(category, clip, tokenizer, encoder, transform, device, pilot, progress):
    import torch
    from PIL import Image
    from torch.nn.functional import normalize
    from expanded_attr_model import ranks_from_scores

    rows, ids, hashes, text_sha = prepare(category, pilot)
    positions = {name: i for i, name in enumerate(ids)}
    targets = torch.tensor([positions[r['target']] for r in rows], device=device)
    refs = torch.tensor([positions[r['candidate']] for r in rows], device=device)
    path_for = image_paths(category)
    features, image_digests = [], {}
    scores_dtype = torch.bfloat16

    def forward_context():
        return torch.autocast(device, dtype=scores_dtype, enabled=device == 'cuda')

    def text_features(strings):
        with forward_context():
            return normalize(clip.encode_text(tokenizer(strings).to(device)).float(), dim=-1)

    with torch.inference_mode():
        for i in range(0, len(ids), 8):
            inputs = []
            for name in ids[i:i + 8]:
                blob = path_for(name).read_bytes()
                digest = hashlib.sha256(blob).hexdigest()
                if hashes is not None and hashes[name] != digest:
                    raise ValueError(f'Gallery JPEG changed: {name}')
                image_digests[name] = digest
                with Image.open(io.BytesIO(blob)) as image:
                    inputs.append(transform(image.convert('RGB')))
            with forward_context():
                image_features = clip.encode_image(torch.stack(inputs).to(device))
            features.append(normalize(image_features.float(), dim=-1))
            if i % 64 == 0 or i + 8 >= len(ids):
                progress('gallery', min(i + 8, len(ids)), len(ids))
        gallery = torch.cat(features)
        if not torch.isfinite(gallery).all():
            raise FloatingPointError('Nonfinite image feature')

        coverage, truncation = [], dict.fromkeys(SLOTS, 0)
        ranks = {name: [] for name in METHODS}
        for i in range(0, len(rows), 8):
            batch = rows[i:i + 8]
            attrs = [row['target_attrs'] for row in batch]
            mod = text_features([row['modification_text'] for row in batch])
            with forward_context():
                packed = encoder(attrs)
            valid = packed['attr_valid']
            coverage.extend(valid.any(1).tolist())
            for name, count in zip(SLOTS, packed['truncated'].sum(0).tolist()):
                truncation[name] += count
            independent = text_features([f'{k.title()}: {row[k]}' for row in attrs for k in SLOTS])
            independent = independent.reshape(len(batch), len(SLOTS), -1)
            mask = valid[..., None]
            denominator = valid.sum(1, keepdim=True).clamp_min(1)
            pooled = (packed['f_attr_text'] * mask).sum(1) / denominator
            standalone = (independent * mask).sum(1) / denominator
            queries = dict(raw_mod=mod, align_slots=normalize(pooled, dim=-1),
                           align_slots_eos=normalize(packed['f_global_text'] + pooled, dim=-1),
                           native_eos_slots=normalize(standalone, dim=-1))
            target, ref = targets[i:i + len(batch)], refs[i:i + len(batch)]
            legal = target != ref
            for name, query in queries.items():
                query = torch.where(valid.any(1, keepdim=True), query, mod)
                scores = query @ gallery.T
                if not torch.isfinite(scores).all():
                    raise FloatingPointError('Nonfinite retrieval score')
                rank = torch.full_like(target, len(ids))
                rank[legal] = ranks_from_scores(scores[legal], target[legal], ref[legal])
                ranks[name].extend(rank.tolist())
            progress('queries', i + len(batch), len(rows))
    if not torch.equal(encoder.embedding.extra, clip.token_embedding.weight.mean(0).repeat(4, 1)):
        raise ValueError('Untrained alignment rows changed')
    return dict(category=category, text_sha256=text_sha, query_indices=[r['source_index'] for r in rows],
                gallery_ids=ids, gallery_sha256=image_digests, same_ref_target=int((targets == refs).sum()),
                queries=len(rows), gallery=len(ids), valid_attribute_queries=sum(coverage), truncation=truncation,
                rows={key: dict(metrics(value), ranks=value) for key, value in ranks.items()},
                versus_raw={name: {str(k): transitions(value, ranks['raw_mod'], k) for k in (1, 10, 50)}
                            for name, value in ranks.items() if name != 'raw_mod'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--categories', nargs='+', choices=tuple(EXPECTED), default=tuple(EXPECTED))
    args = parser.parse_args()
    args.root.mkdir()
    state_path = args.root / 'state.json'
    state = dict(status='running', identity=identity(), stage='checkpoint', device=args.device,
                 pilot=args.pilot, categories=args.categories, completed_categories=[])
    write(state_path, state)
    try:
        import open_clip
        import torch
        from attr_slot_fusion import MultiAttrTextEncoder
        from safetensors.torch import load_file

        torch.set_num_threads(12 if args.device == 'cpu' else 4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if sha(args.weights) != MODEL_SHA:
            raise ValueError('Public raw OpenCLIP checkpoint SHA mismatch')
        clip, _, transform = open_clip.create_model_and_transforms('ViT-H-14', pretrained=None)
        clip.load_state_dict(load_file(str(args.weights)), strict=True)
        clip = clip.to(args.device).eval()
        tokenizer = open_clip.get_tokenizer('ViT-H-14')
        encoder = MultiAttrTextEncoder(clip, tokenizer, strict_segment_isolation=True).eval()
        encoder.requires_grad_(False)
        if any(p.requires_grad for p in encoder.parameters()):
            raise RuntimeError('Unexpected trainable parameters')
        protocol = dict(checkpoint_sha256=MODEL_SHA, source_sha256=sha(__file__), api_calls=0,
                        fit_updates=0, query_side_image_encoder=False, stored_attributes_can_use_reference=True,
                        methods=METHODS, precision='BF16 CUDA or FP32 CPU with FP32 scoring',
                        token_initialization='four identical frozen vocabulary-mean rows',
                        fallback='raw modification when all slots are missing',
                        pooling='equal valid-slot mean; EOS coefficient 1 in align_slots_eos',
                        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), torch=torch.__version__)
        write(args.root / 'protocol.json', protocol)
        for category in args.categories:
            def progress(stage, completed, total):
                state.update(category=category, stage=stage, completed=completed, total=total)
                write(state_path, state)

            result = run_category(category, clip, tokenizer, encoder, transform, args.device, args.pilot, progress)
            write(args.root / f'{category}.json', result)
            state['completed_categories'].append(category)
            write(state_path, state)
        state.update(status='complete', stage='done')
        write(state_path, state)
    except Exception:
        state.update(status='failed', error=traceback.format_exc())
        write(state_path, state)
        raise


if __name__ == '__main__':
    main()
