"""Frozen queries against the full eligible training image gallery; zero API and fit."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image

import newModel
from newTrain import load_checkpoint, load_dqu_checkpoint
from evaluate_image_keyword_routes import autocast
from experiment_storage import require_space, atomic_torch_save
from fashioniq_noise import file_sha
from fiq_gallery_diagnostic_math import gallery_ids, rank_summary
from fiq_small_math import blend
from generate_structured_edits import atomic_write_json


def prepare(args):
    folder = args.research/'train_only_route_gate'/args.dataset
    old_protocol = json.loads((folder/'protocol.json').read_text())
    old_manifest = json.loads((folder/'manifest.json').read_text())
    expected = old_protocol['evidence']
    source = Path(__file__).resolve().parent
    if (file_sha(source/'newModel.py') != expected['model_source_sha256']
            or file_sha(source/'newDataset.py') != expected['dataset_source_sha256']):
        raise ValueError('Frozen model/dataset sources changed')
    main = json.loads((args.primary/'experiment_report.json').read_text())
    if main['status'] != 'complete':
        raise ValueError('Primary comparison must be complete')
    base = args.primary/args.dataset/'checkpoints'/f'{args.dataset}_dqu_original_seed42_best.pt'
    adapter = args.primary/args.dataset/'checkpoints'/f'{args.dataset}_structured_off_seed42_best.pt'
    if (file_sha(base) != main['checkpoint_hashes'][args.dataset]
            or main['checkpoint_hashes'][args.dataset] != expected['base_sha256']
            or file_sha(adapter) != expected['adapter_sha256']):
        raise ValueError('Protected category weights changed')
    matching = []
    for path in args.query_cache.glob(args.dataset+'_*.pt'):
        cache = torch.load(path, map_location='cpu', weights_only=True)
        if cache['evidence'] == expected:
            matching.append((path, cache))
    if len(matching) != 1:
        raise ValueError('Require one exact verified train-only query cache')
    cache_path, cache = matching[0]
    all_rows = old_manifest['fit']+old_manifest['internal_dev']
    if (cache['source_ids'] != [r['candidate'] for r in all_rows]
            or cache['target_ids'] != [r['target'] for r in all_rows]
            or len(cache['qd']) != len(all_rows) or len(cache['qq']) != len(all_rows)):
        raise ValueError('Cached query row binding changed')
    for name in ('qd','qq','targets'):
        if not torch.isfinite(cache[name]).all():
            raise FloatingPointError('Nonfinite cached features')
    queries = old_manifest['internal_dev']
    if len(queries) != 256:
        raise ValueError('Keep the complete fixed 256-query internal dev')
    heldout, metadata = set(), {}
    for pattern in ('split.*.val.json','split.*.test.json'):
        for path in sorted((args.root/'image_splits').glob(pattern)):
            heldout.update(json.loads(path.read_text()))
            metadata[path.name] = file_sha(path)
    if len(metadata) != 6:
        raise ValueError('Require all three categories of official val and test IDs')
    train = args.root/'image_splits'/f'split.{args.dataset}.train.json'
    metadata[train.name] = file_sha(train)
    ids, counts = gallery_ids(json.loads(train.read_text()), heldout, queries)
    hashes = {}
    for image_id in ids:
        path = args.root/'resized_image'/(image_id+'.jpg')
        if not path.is_file():
            raise FileNotFoundError('Incomplete training gallery; do not silently drop images')
        hashes[image_id] = file_sha(path)
    evidence = dict(category=args.dataset, base_sha256=expected['base_sha256'],
        adapter_sha256=expected['adapter_sha256'], query_cache_sha256=file_sha(cache_path),
        inherited_cache_evidence=expected, metadata_sha256=metadata,
        source_sha256=file_sha(__file__), math_source_sha256=file_sha(source/'fiq_gallery_diagnostic_math.py'),
        model_source_sha256=expected['model_source_sha256'],
        images_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest())
    return dict(base=base, adapter=adapter, ids=ids, queries=queries, cache=cache, nfit=len(old_manifest['fit']),
        evidence=evidence, image_sha256=hashes, counts=counts)


def encode_missing(args, context):
    ids, cache = context['ids'], context['cache']
    signature = hashlib.sha256(json.dumps(context['evidence'],sort_keys=True).encode()).hexdigest()[:16]
    path = args.gallery_cache/f'{args.dataset}_{signature}.pt'
    if path.exists():
        saved = torch.load(path,map_location='cpu',weights_only=True)
        if saved['evidence'] != context['evidence'] or saved['ids'] != ids:
            raise ValueError('Gallery cache evidence mismatch')
        return saved
    require_space(args.gallery_cache,1)
    reused = {}
    wanted = set(ids)
    for index,image_id in enumerate(cache['target_ids']):
        if image_id in wanted:
            reused.setdefault(image_id, cache['targets'][index])
    config = torch.load(context['adapter'],map_location='cpu',weights_only=True)['config']
    if config['dataset'] != args.dataset:
        raise ValueError('Wrong category adapter')
    model = newModel.DQU_CIR(clip_checkpoint=args.clip_checkpoint,freeze_clip=True,
        adapter_rank=int(config.get('adapter_rank',256)),
        max_structured_weight=float(config.get('max_structured_weight',.15)),
        max_residual_norm=float(config.get('max_residual_norm',1))).cuda()
    load_dqu_checkpoint(context['base'],model,args.dataset)
    load_checkpoint(context['adapter'],model,args.dataset)
    model.freeze_dqu_backbone()
    model.eval()
    def encode(batch):
        tensors = []
        for image_id in batch:
            image_path = args.root/'resized_image'/(image_id+'.jpg')
            if file_sha(image_path) != context['image_sha256'][image_id]:
                raise ValueError('Bound image changed while encoding')
            with Image.open(image_path) as image:
                tensors.append(model.preprocess_val(image.convert('RGB')))
        with torch.inference_mode(), autocast(torch.device('cuda')):
            result = model.extract_target(torch.stack(tensors).cuda()).float().cpu()
        if not torch.isfinite(result).all():
            raise FloatingPointError('Nonfinite training gallery features')
        return result
    check_ids = sorted(reused)[:16]
    if not check_ids:
        raise ValueError('No cached train targets available for reproducibility check')
    check = encode(check_ids)
    difference = float((check-torch.stack([reused[i] for i in check_ids])).abs().max())
    if difference > 1e-4:
        raise ValueError('Cached target feature reproduction exceeds fixed1e-4 tolerance')
    missing = [i for i in ids if i not in reused]
    fresh = {}
    for start in range(0,len(missing),16):
        require_space(args.output)
        batch = missing[start:start+16]
        values = encode(batch)
        fresh.update(zip(batch,values))
        if start%256 == 0:
            print(json.dumps(dict(stage='encode_missing_train_gallery',done=start,total=len(missing))),flush=True)
    result = dict(evidence=context['evidence'],ids=ids,
        gallery=torch.stack([reused[i] if i in reused else fresh[i] for i in ids]),
        reused_images=len(reused),encoded_images=len(missing),
        reproduction_check_images=len(check_ids),max_cached_feature_difference=difference)
    args.gallery_cache.mkdir(parents=True,exist_ok=True)
    atomic_torch_save(result,path)
    del model,check,fresh
    gc.collect()
    torch.cuda.empty_cache()
    return result


def evaluate(context,saved):
    cache = context['cache'];n = context['nfit'];queries = context['queries']
    ids = context['ids'];lookup = {image_id:i for i,image_id in enumerate(ids)}
    target = torch.tensor([lookup[q['target']] for q in queries])
    source = [lookup.get(q['candidate']) for q in queries]
    routes = dict(dqu_written=cache['qd'][n:],qwen_add=cache['qq'][n:],
        fixed_dual05=blend(cache['qd'][n:],cache['qq'][n:],.5))
    result,neighbors = {}, []
    for name,feature in routes.items():
        scores = feature @ saved['gallery'].T
        if not torch.isfinite(scores).all():
            raise FloatingPointError('Nonfinite gallery scores')
        for i,index in enumerate(source):
            if index is not None:
                scores[i,index] = -torch.inf
        # Stable score ties follow sorted image IDs, identical for all routes.
        order = torch.argsort(scores,descending=True,stable=True)
        rank = (order == target[:,None]).nonzero(as_tuple=False)[:,1]+1
        if len(rank) != 256:
            raise ValueError('Missing target rank; no query deletion')
        ranks = rank.tolist()
        result[name] = dict(all_internal_dev=rank_summary(ranks),
            prior003_positions0_8=rank_summary(ranks[:8]),
            prior004_positions8_16=rank_summary(ranks[8:16]),ranks=ranks)
        if name == 'fixed_dual05':
            for i,q in enumerate(queries):
                head = order[i,:20].tolist()
                neighbors.append(dict(query_index=i,source_index=q['source_index'],
                    candidate=q['candidate'],target=q['target'],modification_text=q['modification_text'],
                    target_rank=ranks[i],candidates=[dict(image_id=ids[j],similarity=float(scores[i,j])) for j in head]))
    return dict(routes=result, fixed_dual05_neighbors=neighbors,reference_images_removed=sum(i is not None for i in source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('shirt','toptee'),required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--primary',type=Path,required=True)
    parser.add_argument('--research',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--clip-checkpoint',required=True)
    parser.add_argument('--query-cache',type=Path,default=Path('/dev/shm/fiq_gate_features_20261026'))
    parser.add_argument('--gallery-cache',type=Path,default=Path('/dev/shm/fiq_train_gallery_20261026'))
    parser.add_argument('--preflight',action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Inspect existing diagnostic; never overwrite or blindly repeat')
    require_space(args.output)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    context = prepare(args)
    if args.preflight:
        print(json.dumps(dict(status='preflight_passed',counts=context['counts'],queries=len(context['queries']),
            official_val_queries_used=False,api_calls=0,model_training=False)),flush=True)
        return
    args.output.mkdir(parents=True)
    atomic_write_json(args.output/'protocol.json',dict(evidence=context['evidence'],seed=42,
        routes=['dqu_written','qwen_add','fixed_dual05'],query_selection='All256 existing fixed hash-order internal-dev queries',
        gallery_selection='All category official-train IDs, excluding every category official-val/test ID; no gold injection',
        api_calls=0,training=False,max_wall_seconds=3600,
        cache_reproduction_tolerance=1e-4,statistic='top20-to-top1 annotated-target oracle headroom, not measured gain',
        limitation='Frozen DQU already saw these training images; not a backbone-independent holdout or official-val result'))
    atomic_write_json(args.output/'manifest.json',dict(category=args.dataset,queries=context['queries'],
        gallery_ids=context['ids'],image_sha256=context['image_sha256'],counts=context['counts']))
    atomic_write_json(args.output/'state.json',dict(status='encoding_missing_gallery',started_unix=time.time()))
    saved = encode_missing(args,context)
    result = evaluate(context,saved)
    require_space(args.output)
    atomic_write_json(args.output/'results.json',dict(status='complete',category=args.dataset,
        counts=context['counts'],queries=len(context['queries']),
        feature_reuse={k:saved[k] for k in ('reused_images','encoded_images','reproduction_check_images','max_cached_feature_difference')},
        evidence=context['evidence'],**result))
    atomic_write_json(args.output/'state.json',dict(status='complete',finished_unix=time.time(),api_calls=0,training=False))
    print(json.dumps(dict(status='complete',category=args.dataset,counts=context['counts'],
        fixed_dual05=result['routes']['fixed_dual05']['all_internal_dev'])),flush=True)


if __name__ == '__main__':
    main()
