"""JPEG-byte isolation for a new head's supervision; retrieval IDs stay unchanged."""
from collections import Counter, defaultdict

from fiq_candidate_partition import image_bucket


def clean_partition(ids, hashes, fit, dev, heldout_hashes):
    if len(set(ids)) != len(ids) or any(i not in hashes for i in ids):
        raise ValueError('Incomplete or duplicate gallery identity')
    heldout = set(heldout_hashes)
    dev_sha = {hashes[i] for i in ids if image_bucket(i) == 0}
    banks = {
        'fit': [j for j, i in enumerate(ids) if image_bucket(i) != 0
                and hashes[i] not in dev_sha | heldout],
        'dev': [j for j, i in enumerate(ids) if image_bucket(i) == 0
                and hashes[i] not in heldout],
    }
    masks, reasons, active, canonical = {}, {}, {}, {}
    for name, rows in (('fit', fit), ('dev', dev)):
        allowed = set(banks[name])
        allowed_ids = {ids[j] for j in allowed}
        row_reasons = []
        for row in rows:
            why = []
            for role, key in (('reference', 'candidate'), ('target', 'target')):
                i = row[key]
                if i not in hashes:
                    raise ValueError('Missing supervised image byte SHA')
                if hashes[i] in heldout:
                    why.append(role + '_official_heldout_byte')
                if name == 'fit' and hashes[i] in dev_sha:
                    why.append(role + '_dev_bank_byte')
                if i not in allowed_ids and not why:
                    why.append(role + '_outside_bank')
            row_reasons.append(why)
        reasons[name] = row_reasons
        masks[name] = [not why for why in row_reasons]
        active[name] = [j for j, good in enumerate(masks[name]) if good]
        seen = set()
        canonical[name] = []
        # Stable ID order selects a representative for negatives only. Positives keep actual ID.
        for j in banks[name]:
            if hashes[ids[j]] not in seen:
                canonical[name].append(j)
                seen.add(hashes[ids[j]])
    if set(hashes[ids[j]] for j in banks['fit']) & set(hashes[ids[j]] for j in banks['dev']):
        raise ValueError('Fit/dev bank byte overlap')
    return dict(banks=banks, canonical=canonical, masks=masks, active=active, reasons=reasons,
                reason_counts={n: dict(Counter(v for row in rows for v in row)) for n, rows in reasons.items()})


def positive_byte_sets(rows, hashes):
    positives = defaultdict(set)
    for row in rows:
        # ID aliases of the same reference and exact modification share known positives.
        key = (hashes[row['candidate']], row['modification_text'])
        positives[key].add(hashes[row['target']])
    return positives


def verify_actual_inputs(ids, hashes, rows, active, pools, banks, heldout_hashes):
    used = {}
    for name in ('fit', 'dev'):
        allowed = set(banks[name])
        chosen = active[name]
        if len(chosen) != len(pools[name]):
            raise ValueError('Active rows/pool length mismatch')
        index = {i: j for j, i in enumerate(ids)}
        positives = positive_byte_sets(rows[name], hashes)
        used[name] = set()
        for row_index, pool in zip(chosen, pools[name]):
            row = rows[name][row_index]
            if len(pool) != 32 or pool[0] != index[row['target']] or not set(pool) <= allowed:
                raise ValueError('Invalid positive binding or forbidden pool input')
            ref = index[row['candidate']]
            if ref not in allowed:
                raise ValueError('Forbidden supervised reference')
            sha = [hashes[ids[j]] for j in pool]
            if len(set(sha)) != 32:
                raise ValueError('Byte alias incorrectly used as a negative')
            forbidden = positives[(hashes[row['candidate']], row['modification_text'])] | {hashes[row['candidate']]}
            if set(sha[1:]) & forbidden:
                raise ValueError('Reference/known-positive byte alias used as negative')
            used[name].update(sha)
            used[name].add(hashes[row['candidate']])
    intersections = dict(fit_dev=len(used['fit'] & used['dev']),
                         fit_heldout=len(used['fit'] & set(heldout_hashes)),
                         dev_heldout=len(used['dev'] & set(heldout_hashes)))
    if any(intersections.values()):
        raise ValueError('Actual supervised input byte overlap')
    return dict(intersections=intersections, actual_input_unique_sha={n: len(s) for n, s in used.items()},
                all_pools_have_31_unique_visual_negatives=True, positive_and_reference_alias_negatives=0,
                fit_statistics_use_only_eligible_pools=True)


def clean_training_pool(qd, qq, gallery, sources, targets, ids, hashes, rows, active, bank, canonical):
    import torch
    from fiq_small_math import blend

    selected = torch.tensor(active, dtype=torch.long)
    scores = blend(qd[selected], qq[selected], .5) @ gallery.T
    negative = scores.clone()
    allowed = torch.zeros(len(ids), dtype=torch.bool)
    allowed[canonical] = True
    negative[:, ~allowed] = -torch.inf
    by_sha = defaultdict(list)
    for j, i in enumerate(ids):
        by_sha[hashes[i]].append(j)
    positives = positive_byte_sets(rows, hashes)
    for j, original in enumerate(active):
        row = rows[original]
        if int(sources[original]) not in bank or int(targets[original]) not in bank:
            raise ValueError('Supervised row outside cleaned positive bank')
        forbidden = positives[(hashes[row['candidate']], row['modification_text'])] | {hashes[row['candidate']]}
        blocked = [i for sha in forbidden for i in by_sha[sha]]
        negative[j, blocked] = -torch.inf
    hard = torch.argsort(negative, descending=True, stable=True)[:, :31]
    if not torch.isfinite(negative.gather(1, hard)).all():
        raise ValueError('Fewer than31 byte-unique eligible hard negatives')
    pool = torch.cat((targets[selected, None], hard), 1)
    return pool, scores.gather(1, pool)
