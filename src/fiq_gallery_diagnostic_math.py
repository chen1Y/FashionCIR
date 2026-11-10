"""ID filtering and descriptive summaries for a fixed train-gallery diagnostic."""
import math
import statistics


def gallery_ids(train_ids, heldout_ids, queries):
    train, heldout = set(train_ids), set(heldout_ids)
    selected = sorted(train - heldout)
    if not selected:
        raise ValueError('Empty eligible training gallery')
    for query in queries:
        if query['candidate'] in heldout or query['target'] in heldout:
            raise ValueError('Heldout image in a diagnostic query')
        if query['candidate'] == query['target']:
            raise ValueError('Reference equals target')
        if query['target'] not in selected:
            raise ValueError('Missing query target; never inject it or drop the query')
    return selected, dict(raw_train_ids=len(train_ids), unique_train_ids=len(train),
        excluded_heldout_ids=len(train & heldout), eligible_gallery=len(selected))


def rank_summary(ranks):
    if not ranks or any(type(rank) is not int or rank < 1 for rank in ranks):
        raise ValueError('Require all positive integer ranks')
    n = len(ranks)
    ordered = sorted(ranks)
    return dict(queries=n,
        recalls={f'r{k}':100 * sum(r <= k for r in ranks) / n for k in (1,5,10,20,50,100)},
        mrr=sum(1/r for r in ranks)/n, mean_rank=sum(ranks)/n, median_rank=statistics.median(ranks),
        p95_rank=ordered[math.ceil(.95*n)-1], max_rank=max(ranks),
        top20_missed_top1=sum(1 < r <= 20 for r in ranks), outside_top20=sum(r > 20 for r in ranks),
        top20_to_top1_oracle_headroom_pp=100*sum(1 < r <= 20 for r in ranks)/n,
        note='Single annotated-target rank; oracle headroom is an upper bound, not attainable or measured gain.')
