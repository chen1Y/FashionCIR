"""All original FashionIQ train records; byte-connected train/dev partition."""
import json,sqlite3
from pathlib import Path
from expanded_attr_metadata import DATA,OLD,PRIMARY,heldout
from fiq_full_partition_024 import partition
from fiq_byte_clean_pools import positive_byte_sets
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json,load_fashioniq_queries

CATS=('dress','shirt','toptee')
BASES={cat:(Path('/data/8T1/lcy/CIRData/autodl-tmp/checkpoints/dress_dqu_official_original_seed42_best.pt') if cat=='dress' else PRIMARY/cat/'checkpoints'/(cat+'_dqu_original_seed42_best.pt')) for cat in CATS}
DRESS_SHA='a505052a947dfad3b8f1488a66ea316ef3e2688def3c17fae730c80e58cd659f'


def prepare_train(root):
    h=heldout();metadata=root/'metadata';metadata.mkdir(exist_ok=True);cache_hashes=dict(h['image_sha256'])
    for cat in ('shirt','toptee'):
        p=OLD/'train_gallery_diagnostic'/cat/'manifest.json';d=json.loads(p.read_text());summary=json.loads((p.parent.parent/'completion_summary.json').read_text())
        if file_sha(p)!=summary['categories'][cat]['manifest_sha256']:raise ValueError('Gallery provenance changed')
        for i,sha in d['image_sha256'].items():
            if i in cache_hashes and cache_hashes[i]!=sha:raise ValueError('Conflicting existing byte indexes')
            cache_hashes[i]=sha
    reports={}
    for cat in CATS:
        target=metadata/(cat+'_train.json')
        if target.exists():raise FileExistsError('Full metadata is immutable')
        base_sha=file_sha(BASES[cat])
        expected=DRESS_SHA if cat=='dress' else json.loads((OLD/'train_only_route_gate'/cat/'protocol.json').read_text())['evidence']['base_sha256']
        if base_sha!=expected:raise ValueError('Original category checkpoint changed')
        split=DATA/'image_splits'/('split.'+cat+'.train.json');ids=json.loads(split.read_text());hashes={}
        for image_id in ids:
            digest=file_sha(DATA/'resized_image'/(image_id+'.jpg'))
            if image_id in cache_hashes and cache_hashes[image_id]!=digest:raise ValueError('Dataset byte source changed')
            hashes[image_id]=digest;cache_hashes[image_id]=digest
        rows=load_fashioniq_queries(DATA,cat,'train')
        if len(rows)!={'dress':5985,'shirt':5988,'toptee':6027}[cat]:raise ValueError('Original full train count changed')
        policy=partition(ids,hashes,rows,h['image_sha256'].values());records=policy['selected']['fit']+policy['selected']['dev'];n=len(policy['selected']['fit']);dev=len(policy['selected']['dev'])
        if n<128 or dev<32:raise ValueError('Full component split lacks supervised train/dev queries; do not change seed')
        known=positive_byte_sets(rows,hashes)
        payload=dict(category=cat,phase='train',records=records,offsets=dict(fit=list(range(n)),dev=list(range(n,n+dev))),banks=policy['banks'],canonical=policy['canonical'],image_sha256=hashes,known_positive_sha={json.dumps(k):sorted(v) for k,v in known.items()},base_sha256=base_sha,heldout_index_sha256=file_sha(OLD/'train_only_candidate_head/official_heldout_byte_index.json.gz'),original_train_rows=len(rows),all_original_records=rows,supervision_audit=policy['audit'],reason_counts=policy['reason_counts'],component_count=policy['components'],byte_intersections=policy['intersections'],caption_sha256=file_sha(DATA/'captions'/('cap.'+cat+'.train.json')),split_sha256=file_sha(split),partition='train-query JPEG connected components, seed42 hash%4 dev; no row subsampling')
        atomic_write_json(target,payload);reports[cat]=dict(original_rows=len(rows),fit=n,dev=dev,masked=len(rows)-n-dev,gallery=len(ids),fit_gallery=len(policy['banks']['fit']),dev_gallery=len(policy['banks']['dev']),components=policy['components'],reason_counts=policy['reason_counts'],metadata_sha256=file_sha(target))
    atomic_write_json(root/'full_train_inventory.json',dict(categories=reports,original_rows=sum(d['original_rows'] for d in reports.values()),byte_independent_new_learning_inputs=True,backbone_independent_holdout=False));print(json.dumps(reports),flush=True)


def prepare_val(root):
    for cat in CATS:
        for arm in ('original','slots','dual'):
            if not (root/'fits'/cat/arm/'fit_complete.json').exists():raise ValueError('All nine full fits must complete first')
    h=heldout()
    for cat in CATS:
        p=root/'metadata'/(cat+'_val.json')
        if p.exists():raise FileExistsError('Official validation metadata already exists')
        rows=load_fashioniq_queries(DATA,cat,'val');split=DATA/'image_splits'/('split.'+cat+'.val.json');ids=json.loads(split.read_text())
        if len(rows)!={'dress':2017,'shirt':2038,'toptee':1961}[cat]:raise ValueError('Full val query count changed')
        if any(r['candidate'] not in ids or r['target'] not in ids for r in rows):raise ValueError('Missing official reference/GT, never insert')
        atomic_write_json(p,dict(category=cat,phase='val',records=rows,gallery_ids=ids,image_sha256={i:h['image_sha256'][i] for i in ids},base_sha256=file_sha(BASES[cat]),caption_sha256=file_sha(DATA/'captions'/('cap.'+cat+'.val.json')),split_sha256=file_sha(split),all_fits_complete_before_official_evaluation=True))
