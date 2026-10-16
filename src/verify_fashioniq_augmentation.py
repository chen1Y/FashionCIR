"""CPU-only overlay audit using both real dataset adapters, without model weights."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

from fashioniq_augmentation import validate, write_json
from fashioniq_noise import file_sha


def audit(root, overlay):
    root,overlay=Path(root).resolve(),Path(overlay).resolve()
    manifest=json.loads((overlay/'augmentation_manifest.json').read_text())
    ctx,accepted=validate(root,manifest);category=manifest['category']
    original=json.loads((root/f'captions/cap.{category}.train.json').read_text())
    augmented=json.loads((overlay/f'captions/cap.{category}.train.json').read_text())
    assert augmented[:len(original)]==original and len(augmented)==len(original)+len(accepted)
    assert (overlay/'resized_image').resolve()==(root/'resized_image').resolve()
    assert (overlay/'image_splits').resolve()==(root/'image_splits').resolve()
    for source in (root/'captions').glob('cap.*.*.json'):
        if source.name.endswith(('.val.json','.test.json')):
            assert file_sha(source)==file_sha(overlay/'captions'/source.name)
    # Adapter constructors normally materialize all validation images. Substitute only
    # image I/O here; separate real written-image checks below exercise new entries.
    from dquDataset import FashionIQ as Original
    from newDataset import FashionIQ as Structured
    transform=[lambda im:im,lambda im:im]
    checks={}
    for name,cls in [('original_dqu',Original),('structured_fallback',Structured)]:
        with ExitStack() as stack:
            stack.enter_context(patch.object(cls,'get_img',return_value=(None,'metadata-only')))
            stack.enter_context(patch.object(cls,'get_written_img',return_value=(None,'metadata-only')))
            base=cls(str(root),category,transform)
            added=cls(str(overlay),category,transform)
        assert len(base)==len(original) and len(added)==len(augmented)
        assert len(base.test_queries)==len(added.test_queries)
        assert len(base.test_targets)==len(added.test_targets)
        columns=('source_img_id','target_img_id','textual_query')
        assert [tuple(q[k] for k in columns) for q in base.test_queries]==[tuple(q[k] for k in columns) for q in added.test_queries]
        rendered=[]
        for row in accepted[-3:]:
            image,_=added.get_written_img(row['candidate'],row['target'],stage=0)
            assert image.size==(512,512)
            rendered.append(dict(candidate=row['candidate'],target=row['target'],written_image_size=list(image.size)))
        checks[name]=dict(train_before=len(base),train_after=len(added),validation_queries=len(added.test_queries),
                          gallery_images=len(added.test_targets),real_written_image_checks=rendered)
    return dict(status='passed',category=category,original_triplets=len(original),added_triplets=len(accepted),
                total_triplets=len(augmented),added_physical_images=0,training_image_ids=ctx['train_count'],
                validation_and_test_unchanged=True,shared_original_image_directory=True,loaders=checks,
                model_weights_loaded=False,retrieval_training_performed=False,
                note='Image I/O mocked only during full metadata construction; three accepted written images per adapter decoded/rendered for real')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--overlay',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=audit(args.root,args.overlay);write_json(args.output,result);print(json.dumps(result))


if __name__=='__main__':main()
