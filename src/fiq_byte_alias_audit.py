"""Reproduce exact-byte alias statistics; no semantic positive labels are inferred."""
import collections


def audit(category,ids,hashes,fit,dev,pools,heldout_hashes,splits):
    groups=collections.Counter(hashes.values())
    aliases=dict(exact_byte_duplicate_groups=sum(v>1 for v in groups.values()),
        images_in_duplicate_groups=sum(v for v in groups.values() if v>1))
    used={}
    for name,rows in [('fit',fit),('dev',dev)]:
        used[name]={row['candidate'] for row in rows}|{ids[i] for pool in pools[name] for i in pool}
        aliases[name+'_positive_byte_alias_in_negatives']=sum(
            any(hashes[ids[i]]==hashes[row['target']] for i in pool[1:]) for row,pool in zip(rows,pools[name]))
    fit_bytes={hashes[i] for i in used['fit']};dev_bytes={hashes[i] for i in used['dev']}
    held_bytes=set(heldout_hashes.values())
    partition=dict(id_overlap_fit_dev=len(used['fit']&used['dev']),byte_overlap_fit_dev=len(fit_bytes&dev_bytes),
        fit_supervised_ids=len(used['fit']),dev_selection_ids=len(used['dev']))
    for name,rows in [('fit',fit),('dev',dev)]:
        partition[name+'_images_byte_alias_official_heldout']=sum(hashes[i] in held_bytes for i in used[name])
        partition[name+'_queries_reference_or_positive_alias_heldout']=sum(
            any(hashes[row[k]] in held_bytes for k in ('candidate','target')) for row in rows)
        for split in ('val','test'):
            relevant={heldout_hashes[i] for i in splits[f'split.{category}.{split}.json']}
            partition[name+'_images_alias_same_category_'+split]=sum(hashes[i] in relevant for i in used[name])
            partition[name+'_queries_ref_or_gt_alias_same_category_'+split]=sum(
                any(hashes[row[k]] in relevant for k in ('candidate','target')) for row in rows)
    return aliases,partition


def main():
    import argparse,datetime,gzip,json
    from pathlib import Path
    import torch
    from fashioniq_noise import file_sha
    from generate_structured_edits import atomic_write_json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research',type=Path,required=True);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve the previous audit verification')
    folder=args.research/'train_only_candidate_head'
    with gzip.open(folder/'official_heldout_byte_index.json.gz','rt',encoding='utf-8') as stream:index=json.load(stream)
    for name,digest in index['metadata_sha256'].items():
        if file_sha(args.data/'image_splits'/name)!=digest:raise ValueError('Heldout metadata changed since byte indexing')
    splits={name:json.loads((args.data/'image_splits'/name).read_text()) for name in index['metadata_sha256']}
    old_alias=json.loads((folder/'byte_alias_diagnostic.json').read_text())
    old_partition=json.loads((folder/'byte_partition_audit_v2.json').read_text())
    if file_sha(folder/'official_heldout_byte_index.json.gz')!=old_partition['heldout_byte_index_sha256']:
        raise ValueError('Original heldout byte index changed')
    rows={}
    for category in ('shirt','toptee'):
        gallery=json.loads((args.research/'train_gallery_diagnostic'/category/'manifest.json').read_text())
        manifest=json.loads((folder/category/'manifest.json').read_text())
        pools=torch.load(folder/category/'candidate_pools.pt',map_location='cpu',weights_only=True)
        aliases,partition=audit(category,gallery['gallery_ids'],gallery['image_sha256'],manifest['fit'],manifest['internal_dev'],
            {key:value.tolist() for key,value in pools.items()},index['image_sha256'],splits)
        if aliases!=old_alias[category] or partition!=old_partition['categories'][category]:
            raise ValueError('Independent code did not reproduce the original byte-audit counts')
        rows[category]=dict(aliases=aliases,partition=partition)
    atomic_write_json(args.output,dict(status='passed',captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=file_sha(__file__),heldout_index_sha256=file_sha(folder/'official_heldout_byte_index.json.gz'),
        categories=rows,api_calls=0,scope='Recompute counters from the persisted image-byte index and candidate pools; no new training, no benchmark/label changes, no semantic equivalence claim.'))
    print('BYTE_AUDIT_VERIFIED_EXACTLY',flush=True)


if __name__=='__main__':main()
