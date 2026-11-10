from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fiq_candidate_partition import image_bucket
from fiq_unused_query_math import original_partitions,actual_learning_ids,unused_inventory,query_order


class UnusedInventoryTests(unittest.TestCase):
    def row(self,ref='r',target='t',index=0,text='edit'):
        return dict(candidate=ref,target=target,source_index=index,modification_text=text,query_id=str(index))

    def inventory(self,rows,old=(),hashes=None,fit=(),dev=(),held=()):
        hashes=hashes or {'r':'R','t':'T'}
        return unused_inventory(rows,list(old),hashes,set(hashes),set(fit),set(dev),set(held))

    def test_old_query_is_excluded_even_if_source_index_changes(self):
        result=self.inventory([self.row(index=20)],[self.row(index=0)])
        self.assertEqual(result['remaining_count'],0);self.assertEqual(result['overlapping_reason_counts']['already_used_query'],1)

    def test_reference_byte_alias_and_target_selection_alias_are_both_excluded(self):
        result=self.inventory([self.row()],hashes={'r':'old_fit_other_id','t':'old_dev_other_id'},fit={'old_fit_other_id'},dev={'old_dev_other_id'})
        self.assertEqual(result['remaining_count'],0);self.assertEqual(set(result['overlapping_reason_counts']),{'reference_fit_byte','target_selection_byte'})

    def test_visual_self_alias_is_not_treated_as_a_semantic_positive(self):
        result=self.inventory([self.row()],hashes={'r':'SAME','t':'SAME'})
        self.assertEqual(result['remaining_count'],0);self.assertIn('reference_target_same_byte',result['overlapping_reason_counts'])

    def test_duplicate_queries_keep_only_first_fixed_hash_row(self):
        rows=[self.row(index=8),self.row(index=3)];result=self.inventory(rows)
        self.assertEqual(result['remaining_count'],1);self.assertEqual(result['remaining_queries'][0]['source_index'],min(rows,key=query_order)['source_index'])

    def test_official_alias_is_blocked_and_reasons_can_overlap(self):
        result=self.inventory([self.row()],hashes={'r':'H','t':'T'},fit={'H'},held={'H'})
        self.assertEqual(result['remaining_count'],0);self.assertEqual(result['overlapping_reason_counts']['reference_official_byte'],1)
        self.assertEqual(result['overlapping_reason_counts']['reference_fit_byte'],1)

    def test_padding_does_not_wrap_to_last_gallery_image_and_negatives_are_counted(self):
        rows=[self.row('ref','positive')];ids=['positive','negative','padding_last']
        used=actual_learning_ids(ids,rows,[0],[[0,1,-1]],[[True,True,False]])
        self.assertEqual(used,{'ref','positive','negative'});self.assertNotIn('padding_last',used)
        with self.assertRaises(ValueError):actual_learning_ids(ids,rows,[0],[[1,0,-1]],[[True,True,False]])

    def test_original_partition_uses_only_ids_and_fixed_hash_order(self):
        ids=[f'id{i}' for i in range(100)];d=[i for i in ids if image_bucket(i)==0];f=[i for i in ids if image_bucket(i)!=0]
        rows=[self.row(d[0],d[1],2),self.row(f[0],f[1],3),self.row(d[0],f[0],4),self.row(d[2],d[3],5)]
        fit,dev,counts=original_partitions(rows,{d[2]});self.assertEqual(counts,dict(available_fit=1,available_dev=1,official_heldout=1,cross_partition=1))
        self.assertEqual(fit[0]['source_index'],3);self.assertEqual(dev[0]['source_index'],2)

    def test_structured_validity_or_rank_cannot_select_queries(self):
        row=self.row();row.update(valid=False,rank=99999,score=-1000)
        result=self.inventory([row]);self.assertEqual(result['remaining_count'],1);self.assertFalse(result['meets_minimum32'])
        self.assertTrue(all(v==0 for v in result['proof'].values()))


if __name__=='__main__':unittest.main()
