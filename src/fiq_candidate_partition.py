"""Image-level partition shared with the original train-only route-gate trial."""
import hashlib


def image_bucket(image_id):
    return int(hashlib.sha256(f'42|{image_id}'.encode()).hexdigest()[:8],16)%4


def validate_partitions(fit,dev,gallery_ids,heldout):
    if len(set(gallery_ids))!=len(gallery_ids) or set(gallery_ids)&set(heldout):
        raise ValueError('Duplicate or official-heldout image in training gallery')
    fit_bank=[i for i,image_id in enumerate(gallery_ids) if image_bucket(image_id)!=0]
    dev_bank=[i for i,image_id in enumerate(gallery_ids) if image_bucket(image_id)==0]
    gallery=set(gallery_ids)
    for rows,is_dev in ((fit,False),(dev,True)):
        for row in rows:
            if row['candidate']==row['target']:
                raise ValueError('Reference equals positive target')
            for image_id in (row['candidate'],row['target']):
                if image_id in heldout or image_id not in gallery or (image_bucket(image_id)==0)!=is_dev:
                    raise ValueError('Query or positive violates image partition')
    if min(len(fit_bank),len(dev_bank))<33:
        raise ValueError('Insufficient partition-specific negative gallery')
    return fit_bank,dev_bank
