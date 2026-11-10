"""Metadata-only fixed batch geometry for missing global/patch forwards."""


def schedule(old_ids,requested_ids,needed_patch_ids,old_patch_ids):
    if len(set(old_ids))!=len(old_ids) or len(set(requested_ids))!=len(requested_ids):raise ValueError('Duplicate cache/gallery IDs')
    wanted=set(requested_ids);old=set(old_ids);needed=set(needed_patch_ids);patches=set(old_patch_ids)
    if not needed<=wanted or not patches<=old:raise ValueError('Patch provenance is outside its gallery')
    pending=[i for i in requested_ids if i not in old or (i in needed and i not in patches)]
    tail=old_ids[len(old_ids)//8*8:] if len(old_ids)%8 else []
    tailset=set(tail);tail_needed=[i for i in pending if i in tailset];batches=[]
    # Check the known fragile tail first; retain its full original shape/order.
    if tail_needed:batches.append(dict(mode='original_tail',ids=list(tail),real_ids=tail_needed))
    normal=[i for i in pending if i not in tailset]
    for j in range(0,len(normal),8):
        real=normal[j:j+8];batches.append(dict(mode='batch8',ids=real+[real[0]]*(8-len(real)),real_ids=real))
    published=[i for b in batches for i in b['real_ids']]
    if len(published)!=len(set(published)) or set(published)!=set(pending):raise ValueError('Missing/duplicate scheduled real ID')
    original_size={i:(len(tail) if i in tailset else 8) for i in old_ids}
    for batch in batches:
        if any(i in old and original_size[i]!=len(batch['ids']) for i in batch['ids']):raise ValueError('Reused image moved to a different batch shape')
        batch['real_positions']=[batch['ids'].index(i) for i in batch['real_ids']]
    forwards=sum(len(b['ids']) for b in batches)
    return dict(pending_ids=pending,batches=batches,old_tail_ids=tail,counts=dict(missing_global=sum(i not in old for i in pending),missing_patch_for_existing_global=sum(i in old for i in pending),real_images=len(pending),image_forwards=forwards,padding_or_tail_companions=forwards-len(pending)))
