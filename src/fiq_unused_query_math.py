"""Rank-independent inventory of original train queries unused by current head selection."""
from collections import Counter
import hashlib

from fiq_candidate_partition import image_bucket


def query_key(row):
    return row['candidate'],row['target'],row['modification_text']


def query_order(row):
    return hashlib.sha256(f"42|{row['candidate']}|{row['target']}|{row['source_index']}".encode()).hexdigest()


def original_partitions(rows,heldout_ids):
    fit,dev=[],[];dropped=dict(official_heldout=0,cross_partition=0)
    for row in rows:
        ref,target=row['candidate'],row['target']
        if ref in heldout_ids or target in heldout_ids:dropped['official_heldout']+=1
        elif (image_bucket(ref)==0)!=(image_bucket(target)==0):dropped['cross_partition']+=1
        else:(dev if image_bucket(ref)==0 else fit).append(row)
    return sorted(fit,key=query_order),sorted(dev,key=query_order),dict(available_fit=len(fit),available_dev=len(dev),**dropped)


def actual_learning_ids(gallery_ids,rows,selected,indices,masks):
    used=set()
    if len(selected)!=len(indices) or len(indices)!=len(masks):raise ValueError('Pool row mapping mismatch')
    for original,pool,mask in zip(selected,indices,masks):
        if not 0<=original<len(rows) or len(pool)!=len(mask) or not mask[0]:raise ValueError('Invalid supervision row/mask')
        real=[]
        for value,active in zip(pool,mask):
            if not active:
                if value!=-1:raise ValueError('Noncanonical padding index')
                continue
            if not 0<=value<len(gallery_ids):raise ValueError('Invalid real pool index')
            real.append(gallery_ids[value])
        if len(real)<2 or real[0]!=rows[original]['target']:raise ValueError('Invalid actual positive binding')
        used.update(real);used.add(rows[original]['candidate'])
    return used


def unused_inventory(dev_rows,old_rows,hashes,existing_ids,fit_sha,selection_sha,heldout_sha):
    old={query_key(row) for row in old_rows};seen=set();review=[];remaining=[]
    stages=[('unused_query',{'already_used_query','duplicate_dev_query'}),
        ('gallery_and_file_available',{'reference_not_in_gallery','target_not_in_gallery','reference_missing_file','target_missing_file'}),
        ('official_byte_clear',{'reference_official_byte','target_official_byte'}),
        ('current_learning_byte_clear',{'reference_fit_byte','target_fit_byte','reference_selection_byte','target_selection_byte'}),
        ('different_reference_target_bytes',{'reference_target_same_byte'})]
    for row in sorted(dev_rows,key=query_order):
        key=query_key(row);why=[]
        if key in old:why.append('already_used_query')
        if key in seen:why.append('duplicate_dev_query')
        seen.add(key)
        for role,field in [('reference','candidate'),('target','target')]:
            image_id=row[field]
            if image_id not in hashes:why.append(role+'_not_in_gallery')
            if image_id not in existing_ids:why.append(role+'_missing_file')
            if image_id in hashes:
                sha=hashes[image_id]
                if sha in heldout_sha:why.append(role+'_official_byte')
                if sha in fit_sha:why.append(role+'_fit_byte')
                if sha in selection_sha:why.append(role+'_selection_byte')
        if row['candidate'] in hashes and row['target'] in hashes and hashes[row['candidate']]==hashes[row['target']]:
            why.append('reference_target_same_byte')
        review.append(dict(source_index=row['source_index'],query_id=row.get('query_id'),reasons=why))
        if not why:remaining.append(dict(row,reference_sha256=hashes[row['candidate']],target_sha256=hashes[row['target']]))
    cumulative=set();counts={'original_dev':len(dev_rows)}
    for name,reasons in stages:
        cumulative.update(reasons);counts[name]=sum(not (set(r['reasons'])&cumulative) for r in review)
    byte_set={row[k] for row in remaining for k in ('reference_sha256','target_sha256')}
    proof=dict(overlap_fit_sha=len(byte_set&fit_sha),overlap_selection_sha=len(byte_set&selection_sha),overlap_official_sha=len(byte_set&heldout_sha),
        old_query_key_overlap=len({query_key(row) for row in remaining}&old),duplicate_remaining_queries=len(remaining)-len({query_key(row) for row in remaining}))
    if any(proof.values()):raise ValueError('Unused inventory isolation failed')
    return dict(stage_counts=counts,overlapping_reason_counts=dict(Counter(reason for row in review for reason in row['reasons'])),
        reviewed_queries=review,remaining_queries=remaining,proof=proof,remaining_count=len(remaining),meets_minimum32=len(remaining)>=32)
