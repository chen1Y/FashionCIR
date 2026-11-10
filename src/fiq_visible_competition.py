"""Compare saved head order with actual loss candidates; no embeddings or fitting."""
from collections import Counter
import math


def analyze(detail,pool,hashes,allowed,known,reference):
    ids=detail['top20_ids'];scores=detail['top20_scores'];gt=detail['target_id']
    if len(ids)!=len(scores) or len(ids)<2 or len(set(ids))!=len(ids) or gt not in ids:raise ValueError('Invalid head/GT')
    if any(type(s) not in (int,float) or not math.isfinite(s) for s in scores) or any(a<b for a,b in zip(scores,scores[1:])):raise ValueError('Invalid frozen scores/order')
    rank=ids.index(gt)+1
    if rank!=detail['baseline_rank']:raise ValueError('GT rank changed')
    if len(pool)<2 or pool[0]!=gt or len(set(pool))!=len(pool) or not set(pool)<=set(ids):raise ValueError('GT binding/injection/duplicate')
    if any(i not in hashes for i in ids+[reference]):raise ValueError('Missing SHA')
    if not set(pool+[reference])<=allowed or len({hashes[i] for i in pool})!=len(pool):raise ValueError('Forbidden/alias input')
    if hashes[gt] not in known:raise ValueError('Missing known GT')
    seen=set(known)|{hashes[reference]};expected=[]
    for i in ids:
        if i in allowed and hashes[i] not in seen:expected.append(i);seen.add(hashes[i])
    if pool!=[gt]+expected:raise ValueError('Pool does not reproduce legal intersection')
    pset=set(pool);psha={hashes[i] for i in pool};nsha={hashes[i] for i in pool[1:]};ahead=ids[:rank-1];kept=[i for i in ahead if i in pset];values=dict(zip(ids,scores));strong=next(i for i in ids if i!=gt);candidates=[]
    for i in ids:
        reasons=[]
        if i not in pset:
            if i not in allowed:reasons.append('outside_009_bank')
            if hashes[i] in known:reasons.append('known_query_positive_byte')
            if hashes[i]==hashes[reference]:reasons.append('reference_byte')
            if hashes[i] in nsha:reasons.append('retained_negative_byte_alias')
            if not reasons:raise ValueError('Unexplained exclusion')
        candidates.append(dict(image_id=i,sha256=hashes[i],score=values[i],in_loss_by_id=i in pset,in_loss_by_sha=hashes[i] in psha,in_negative_pool_by_sha=hashes[i] in nsha,exclusion_reasons=reasons))
    return dict(original_query_row=detail['original_query_row'],target_id=gt,reference_id=reference,full_gt_rank=rank,visible_gt_rank=1+len(kept),
        hidden_competition=rank>1 and not kept,full_stronger_ids=ahead,visible_stronger_ids=kept,pool_ids=pool,pool_size=len(pool),
        strongest_non_gt_id=strong,strongest_id_covered=strong in pset,strongest_sha_covered=hashes[strong] in psha,strongest_negative_sha_covered=hashes[strong] in nsha,
        strongest_exclusion_reasons=next(c['exclusion_reasons'] for c in candidates if c['image_id']==strong),
        full_margin=values[gt]-values[strong],visible_margin=values[gt]-max(values[i] for i in pool[1:]),candidates=candidates)


def summarize(rows):
    n=len(rows);wrong=sum(r['full_gt_rank']>1 for r in rows);hidden=sum(r['hidden_competition'] for r in rows)
    def stats(key):
        v=[r[key] for r in rows];return dict(mean=sum(v)/n if n else None,minimum=min(v) if n else None,maximum=max(v) if n else None)
    return dict(queries=n,full_non_top1=wrong,full_non_top1_visible_top1=hidden,hidden_fraction_all=hidden/n if n else None,hidden_fraction_non_top1=hidden/wrong if wrong else None,
        strongest_id_covered=sum(r['strongest_id_covered'] for r in rows),strongest_sha_covered=sum(r['strongest_sha_covered'] for r in rows),strongest_negative_sha_covered=sum(r['strongest_negative_sha_covered'] for r in rows),
        strongest_omission_overlapping_reasons=dict(Counter(x for r in rows for x in r['strongest_exclusion_reasons'])),pool_size_distribution=dict(sorted(Counter(r['pool_size'] for r in rows).items())),
        full_stronger_total=sum(len(r['full_stronger_ids']) for r in rows),visible_stronger_total=sum(len(r['visible_stronger_ids']) for r in rows),full_margin=stats('full_margin'),visible_margin=stats('visible_margin'))
