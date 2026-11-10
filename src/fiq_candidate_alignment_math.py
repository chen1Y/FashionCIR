"""Diagnostics of fixed head-only ranking bounds; no fitted parameter or semantic GT."""
import math


def analyze_row(top_ids, top_scores, target_id, baseline_rank, pool_ids, hashes,
                eligible_ids, forbidden_negative_sha, valid, bound=.03):
    if len(top_ids) != 20 or len(top_scores) != 20 or len(set(top_ids)) != 20:
        raise ValueError('Require exactly20 unique frozen head IDs')
    if not all(math.isfinite(x) for x in top_scores) or not 0 < bound < 1:
        raise ValueError('Nonfinite scores/invalid bound')
    inside = target_id in top_ids
    if inside != (baseline_rank <= 20) or (inside and top_ids[baseline_rank-1] != target_id):
        raise ValueError('Target rank/head identity mismatch')
    if len(pool_ids) != 32 or pool_ids[0] != target_id:
        raise ValueError('Require actual positive+31 inherited loss candidates')
    best = worst = baseline_rank
    if inside and valid:
        position = top_ids.index(target_id)
        def counterfactual(sign):
            scores = [value + sign*bound if j == position else value - sign*bound
                      for j, value in enumerate(top_scores)]
            order = sorted(range(20), key=lambda j: (-scores[j], j))
            return order.index(position)+1
        best, worst = counterfactual(1), counterfactual(-1)
    pool_sha = {hashes[i] for i in pool_ids[1:]}
    eligible_negative = []
    seen = set(forbidden_negative_sha)
    for image_id in top_ids:
        sha = hashes[image_id]
        if image_id in eligible_ids and sha not in seen:
            eligible_negative.append(image_id)
            seen.add(sha)
    return dict(target_id=target_id, baseline_rank=baseline_rank,
        bucket='top1' if baseline_rank == 1 else 'rank2_20' if inside else 'outside20',
        structured_valid=bool(valid), closed_bound_best_rank=best, closed_bound_worst_rank=worst,
        top1_newly_reachable=bool(baseline_rank > 1 and best == 1),
        top10_newly_reachable=bool(baseline_rank > 10 and best <= 10),
        top1_can_be_lost=bool(baseline_rank == 1 and worst > 1),
        top10_can_be_lost=bool(baseline_rank <= 10 and worst > 10),
        loss_negative_id_overlap_top20=sum(i in top_ids for i in pool_ids[1:]),
        loss_negative_sha_overlap_top20=len(pool_sha & {hashes[i] for i in top_ids}),
        inference_eligible_unique_negative_count=len(eligible_negative),
        inference_eligible_unique_negative_ids=eligible_negative,
        gt_present_in_inference=inside,
        top20_ids=top_ids, top20_scores=top_scores)


def summarize(rows):
    if not rows:
        raise ValueError('No eligible queries')
    n = len(rows)
    counts = {key: sum(row['bucket'] == key for row in rows) for key in ('top1', 'rank2_20', 'outside20')}
    return dict(queries=n, rank_buckets=counts, rank_bucket_percent={k: 100*v/n for k, v in counts.items()},
        valid_queries=sum(r['structured_valid'] for r in rows),
        forced_gt_outside_inference_top20=counts['outside20'],
        closed_bound_baseline_r1=100*counts['top1']/n,
        closed_bound_optimistic_r1_upper=100*sum(r['closed_bound_best_rank'] <= 1 for r in rows)/n,
        closed_bound_baseline_r10=100*sum(r['baseline_rank'] <= 10 for r in rows)/n,
        closed_bound_optimistic_r10_upper=100*sum(r['closed_bound_best_rank'] <= 10 for r in rows)/n,
        newly_reachable_r1_count=sum(r['top1_newly_reachable'] for r in rows),
        newly_reachable_r10_count=sum(r['top10_newly_reachable'] for r in rows),
        baseline_r1_at_risk_count=sum(r['top1_can_be_lost'] for r in rows),
        baseline_r10_at_risk_count=sum(r['top10_can_be_lost'] for r in rows),
        loss_negative_id_overlap_top20_mean=sum(r['loss_negative_id_overlap_top20'] for r in rows)/n,
        loss_negative_sha_overlap_top20_mean=sum(r['loss_negative_sha_overlap_top20'] for r in rows)/n,
        inference_eligible_negative_mean=sum(r['inference_eligible_unique_negative_count'] for r in rows)/n,
        valid_gt_rank2_20_with_any_eligible_negative=sum(r['structured_valid'] and r['bucket']=='rank2_20'
            and r['inference_eligible_unique_negative_count'] > 0 for r in rows))
