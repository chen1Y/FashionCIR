"""014: metadata/pool-only feasibility inventory, no model encoding, fit, API or retrieval metrics."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from experiment_storage import require_space
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json,load_fashioniq_queries
from fiq_unused_query_math import original_partitions,actual_learning_ids,unused_inventory,query_key


def prepare_inventory(args):
    import torch
    root=args.research
    old_folder=root/'train_only_route_gate'/args.dataset
    old=json.loads((old_folder/'manifest.json').read_text())
    expected=json.loads((old_folder/'protocol.json').read_text())['evidence']
    if hashlib.sha256(json.dumps(old,sort_keys=True).encode()).hexdigest()!=expected['manifest_sha256']:
        raise ValueError('Original002 manifest changed')
    held_path=root/'train_only_candidate_head/official_heldout_byte_index.json.gz'
    held=json.loads(gzip.decompress(held_path.read_bytes()))
    audit=json.loads((root/'train_only_candidate_head/byte_partition_audit_v2.json').read_text())
    if file_sha(held_path)!=audit['heldout_byte_index_sha256'] or len(held['image_sha256'])!=30566:
        raise ValueError('Heldout byte provenance changed')
    held_ids=set()
    for name,digest in held['metadata_sha256'].items():
        path=args.root/'image_splits'/name
        if file_sha(path)!=digest or old['heldout_metadata_sha256'][name]!=digest:raise ValueError('Heldout split metadata changed')
        held_ids.update(json.loads(path.read_text()))
    if held_ids!=set(held['image_sha256']):raise ValueError('Heldout index identity mismatch')
    rows=load_fashioniq_queries(args.root,args.dataset,'train')
    if len(rows)!={'shirt':5988,'toptee':6027}[args.dataset]:raise ValueError('Train count changed')
    structured_path=args.primary/args.dataset/f'structured_{args.dataset}_train.json'
    if file_sha(structured_path)!=expected['structured_train_sha256']:raise ValueError('Original completed text artifact changed')
    structured=json.loads(structured_path.read_text())
    fingerprint=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    if structured['query_sha256']!=fingerprint or structured['state']!='complete' or len(structured['samples'])!=len(rows):
        raise ValueError('Full original train query fingerprint changed')
    samples={sample['source_index']:sample for sample in structured['samples']}
    if set(samples)!=set(range(len(rows))):raise ValueError('Completed sample indices changed')
    for row in rows:
        sample=samples[row['source_index']]
        if any(sample[k]!=row[k] for k in ('candidate','target','modification_text','category','split')):
            raise ValueError('Stored online query binding changed')
    fit,dev,partition_counts=original_partitions(rows,held_ids)
    if partition_counts!=old['partition_counts'] or fit[:1024]!=old['fit'] or dev[:256]!=old['internal_dev']:
        raise ValueError('Original seed42 fit/dev manifest did not reproduce')
    if len(dev)!={'shirt':327,'toptee':315}[args.dataset]:raise ValueError('Available original dev count changed')
    gallery_path=root/'train_gallery_diagnostic'/args.dataset/'manifest.json'
    proof005=json.loads((root/'train_gallery_diagnostic/completion_summary.json').read_text())['categories'][args.dataset]
    if file_sha(gallery_path)!=proof005['manifest_sha256']:raise ValueError('Full train byte manifest changed')
    gallery=json.loads(gallery_path.read_text());ids=gallery['gallery_ids'];hashes=gallery['image_sha256']
    source005=json.loads((root/'train_gallery_diagnostic'/args.dataset/'protocol.json').read_text())['evidence']
    for name,digest in source005['metadata_sha256'].items():
        if file_sha(args.root/'image_splits'/name)!=digest:raise ValueError('Gallery metadata changed')
    folder=root/'supported_candidate_head_011'/args.dataset
    summary011=json.loads((folder.parent/'completion_summary.json').read_text())['categories'][args.dataset]
    current=json.loads((folder/'manifest.json').read_text())
    if file_sha(folder/'manifest.json')!=summary011['manifest_sha256']:raise ValueError('011 supervised manifest changed')
    if file_sha(folder/'head.pt')!=summary011['head_sha256']:raise ValueError('Current011 checkpoint changed')
    artifact=json.loads((folder.parent/'artifact_sha256.json').read_text())
    pool_sha=file_sha(folder/'candidate_pools.pt')
    if pool_sha!=artifact[str((folder/'candidate_pools.pt').relative_to(root))]:raise ValueError('011 pool changed')
    pools=torch.load(folder/'candidate_pools.pt',map_location='cpu',weights_only=True)
    all_original=current['original_fit']+current['original_dev']
    if all_original!=old['fit']+old['internal_dev']:raise ValueError('Current/original row lineage mismatch')
    used_ids={};used_sha={}
    for name,pool in pools.items():
        selected=pool['original_rows'].tolist()
        if selected!=current['selected_original_rows'][name]:raise ValueError('011 selected row mapping changed')
        for indices,mask,images in zip(pool['indices'],pool['mask'],current['candidate_ids'][name]):
            if [ids[i] for i in indices[mask].tolist()]!=images:raise ValueError('PT/machine-readable actual input IDs differ')
        used_ids[name]=actual_learning_ids(ids,all_original,selected,pool['indices'].tolist(),pool['mask'].tolist())
        if not used_ids[name]<=set(hashes):raise ValueError('Unknown learning input byte SHA')
        used_sha[name]={hashes[i] for i in used_ids[name]}
        if len(used_sha[name])!=summary011['isolation_proof']['actual_input_unique_sha'][name]:raise ValueError('Actual learning input SHA count changed')
    held_sha=set(held['image_sha256'].values())
    if used_sha['fit']&used_sha['dev'] or (used_sha['fit']|used_sha['dev'])&held_sha:
        raise ValueError('Existing current learning isolation changed')
    lineage012=json.loads((root/'rank4_candidate_head_012/completion_summary.json').read_text())['categories'][args.dataset]['arms']
    for arm,value in lineage012.items():
        if value['source_pool_sha256']!=pool_sha or value['training']['fit_original_rows']!=current['selected_original_rows']['fit']:
            raise ValueError('012 no longer uses identical011 support')
        if file_sha(root/'rank4_candidate_head_012'/arm/args.dataset/'results.json')!=value['results_sha256']:
            raise ValueError('012 lineage artifact changed')
        if file_sha(root/'rank4_candidate_head_012'/arm/args.dataset/'head.pt')!=value['head_sha256']:
            raise ValueError('Current012 checkpoint changed')
    dev_ids={row[k] for row in dev for k in ('candidate','target')};existing=set();checked={}
    for image_id in sorted(dev_ids):
        path=args.root/'resized_image'/(image_id+'.jpg')
        if path.is_file():
            existing.add(image_id)
            if image_id in hashes:
                digest=file_sha(path)
                if digest!=hashes[image_id]:raise ValueError('Original dev image bytes changed since005')
                checked[image_id]=digest
    evidence=dict(category=args.dataset,source_sha256=file_sha(__file__),math_sha256=file_sha(Path(__file__).with_name('fiq_unused_query_math.py')),
        original_train_query_fingerprint=fingerprint,raw_train_captions_sha256=file_sha(args.root/'captions'/f'cap.{args.dataset}.train.json'),
        original002_manifest_sha256=file_sha(old_folder/'manifest.json'),gallery005_manifest_sha256=file_sha(gallery_path),
        heldout_index_sha256=file_sha(held_path),current011_manifest_sha256=file_sha(folder/'manifest.json'),current011_pool_sha256=pool_sha,
        completed_structured_train_sha256=file_sha(structured_path),checked_dev_image_sha256=checked)
    return dict(rows=rows,dev=dev,partition_counts=partition_counts,old=old,hashes=hashes,existing=existing,
        fit_sha=used_sha['fit'],selection_sha=used_sha['dev'],heldout_sha=held_sha,evidence=evidence,samples=samples,
        actual_learning_inputs={n:dict(unique_ids=len(used_ids[n]),unique_sha=len(used_sha[n])) for n in used_ids})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=('shirt','toptee'),required=True)
    for name in ('root','primary','research','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve014 output; no blind repeat')
    require_space(args.output);c=prepare_inventory(args)
    if args.preflight:
        print(json.dumps(dict(status='preflight_passed',train_queries=len(c['rows']),available_original_dev=len(c['dev']),original002_rows_exact=True,
            actual_learning_inputs=c['actual_learning_inputs'],checked_dev_images=len(c['evidence']['checked_dev_image_sha256']),
            model_loaded=False,official_val_queries_loaded=0,api_calls=0)),flush=True);return
    args.output.mkdir(parents=True)
    atomic_write_json(args.output/'protocol.json',dict(evidence=c['evidence'],api_calls=0,training=False,model_loading=False,feature_encoding=False,
        query_selection='All original dev hash partition, originalseed42 query order; exclude originalfit1024/dev256 exact query keys independent of source_index.',
        exclusions='Either ref or GT SHA in actual011 fit/statistics or devNCE inputs, officialheldout SHA, missing/notingallery, visual self-alias. No score/rank/structuredvalid filter.',
        feasibility_minimum_per_category=32,reservation='Reserve only if>=32; no encoding/evaluation or relaxed threshold in014.',
        scope='Current011/012 head-learning/selection input isolation, not unseen-backbone or whole-research independence. Full galleries remain shared.'))
    result=unused_inventory(c['dev'],c['old']['fit']+c['old']['internal_dev'],c['hashes'],c['existing'],c['fit_sha'],c['selection_sha'],c['heldout_sha'])
    coverage=dict(completed_samples=len(result['remaining_queries']),valid_samples=0,invalid_dqu_fallback=0,queries_in_original002_feature_cache=0)
    old_keys={query_key(row) for row in c['old']['fit']+c['old']['internal_dev']}
    for row in result['remaining_queries']:
        sample=c['samples'][row['source_index']]
        valid=(sample.get('status')=='ok' and sample.get('validation',{}).get('valid') is True and sample.get('semantic_validation',{}).get('valid') is True)
        coverage['valid_samples']+=int(valid);coverage['invalid_dqu_fallback']+=int(not valid)
        coverage['queries_in_original002_feature_cache']+=int(query_key(row) in old_keys)
    result.update(status='complete',category=args.dataset,evidence=c['evidence'],partition_counts=c['partition_counts'],
        actual_learning_inputs=c['actual_learning_inputs'],structured_coverage_reporting_only=coverage,
        evaluation_performed=False,metrics_read_for_selection=False,api_calls=0,model_loaded=False)
    if result['meets_minimum32']:
        atomic_write_json(args.output/'reserved_holdout_manifest.json',dict(category=args.dataset,evidence=c['evidence'],queries=result['remaining_queries'],
            proof=result['proof'],reserved_for_future_frozen_eval=True,evaluation_performed=False,scope='Head-learning and NCE-input isolation only; backbone saw train.'))
    require_space(args.output);atomic_write_json(args.output/'results.json',result);atomic_write_json(args.output/'state.json',dict(status='complete',feasible=result['meets_minimum32']))
    print(json.dumps({k:result[k] for k in ('status','category','remaining_count','meets_minimum32','stage_counts','overlapping_reason_counts','proof','structured_coverage_reporting_only')}),flush=True)


if __name__=='__main__':main()
