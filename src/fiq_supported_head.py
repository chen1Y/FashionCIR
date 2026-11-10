"""Real top20 candidate supervision with explicit padding exclusion."""


def select_supported_rows(details, hashes, allowed_ids, known_positives, source_ids, modifications):
    selected, reasons, pools = [], {}, []
    for row in details:
        original = row['original_query_row']
        source = source_ids[original]
        target = row['target_id']
        forbidden = known_positives[(hashes[source], modifications[original])] | {hashes[source]}
        seen = set(forbidden)
        negatives = []
        for image_id in row['top20_ids']:
            sha = hashes[image_id]
            if image_id in allowed_ids and sha not in seen:
                negatives.append(image_id)
                seen.add(sha)
        if negatives != row['inference_eligible_unique_negative_ids']:
            raise ValueError('010 eligible head candidates no longer reproduce')
        why = []
        if not row['structured_valid']:
            why.append('invalid_structured')
        if target not in row['top20_ids']:
            why.append('gt_outside_frozen_top20')
        if not negatives:
            why.append('no_byte_eligible_head_negative')
        reasons[str(original)] = why
        if why:
            continue
        if source not in allowed_ids or target not in allowed_ids:
            raise ValueError('Forbidden reference/positive supervision')
        pool = [target] + negatives
        if not set(pool) <= set(row['top20_ids']) or not 2 <= len(pool) <= 20:
            raise ValueError('GT injection or invalid supported pool')
        if len({hashes[i] for i in pool}) != len(pool):
            raise ValueError('Byte alias used as negative')
        selected.append(original)
        pools.append(pool)
    return selected, pools, reasons


def masked_loss(head, pool, indices=None):
    import torch
    from torch.nn import functional as F
    x, base, valid, mask = (pool[k] for k in ('x', 'base', 'valid', 'mask'))
    if indices is not None:
        x, base, valid, mask = x[indices], base[indices], valid[indices], mask[indices]
    if not mask[:, 0].all() or not (mask.sum(1) >= 2).all():
        raise ValueError('Every supported row needs a real positive and negative')
    # Sanitizing before the network also prevents NaN padding from contaminating gradients.
    correction = head(torch.where(mask[..., None], x, torch.zeros_like(x)), valid)
    logits = torch.where(mask, (base+correction)/.07, torch.full_like(base, -torch.inf))
    anchor = correction[mask].square().mean()
    return F.cross_entropy(logits, torch.zeros(len(x), dtype=torch.long)) + .01*anchor
