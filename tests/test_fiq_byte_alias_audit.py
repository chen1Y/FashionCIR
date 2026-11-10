from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fiq_byte_alias_audit import audit


class ByteAliasTests(unittest.TestCase):
    def inputs(self):
        ids=['r_fit','gt_fit','alias_gt','r_dev','gt_dev','alias_fit_in_dev']
        hashes=dict(zip(ids,['a','b','b','d','e','b']))
        fit=[dict(candidate='r_fit',target='gt_fit')];dev=[dict(candidate='r_dev',target='gt_dev')]
        pools=dict(fit=[[1,2]],dev=[[4,5]])
        held=dict(val_id='b',test_id='e');splits={'split.shirt.val.json':['val_id'],'split.shirt.test.json':['test_id']}
        return ids,hashes,fit,dev,pools,held,splits

    def test_distinct_ids_can_overlap_bytes_and_conflict_with_positive(self):
        aliases,partition=audit('shirt',*self.inputs())
        self.assertEqual(aliases['fit_positive_byte_alias_in_negatives'],1)
        self.assertEqual(partition['id_overlap_fit_dev'],0)
        self.assertEqual(partition['byte_overlap_fit_dev'],1)

    def test_same_category_heldout_counts_are_separate_from_global(self):
        _,partition=audit('shirt',*self.inputs())
        self.assertEqual(partition['fit_images_alias_same_category_val'],2)
        self.assertEqual(partition['dev_queries_ref_or_gt_alias_same_category_test'],1)

    def test_unequal_hashes_are_not_promoted_to_equivalent_images(self):
        values=list(self.inputs());values[1]={image_id:image_id for image_id in values[0]}
        aliases,partition=audit('shirt',*values)
        self.assertEqual(aliases['exact_byte_duplicate_groups'],0)
        self.assertEqual(aliases['fit_positive_byte_alias_in_negatives'],0)
        self.assertEqual(partition['byte_overlap_fit_dev'],0)


if __name__=='__main__':unittest.main()
