from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fiq_visible_competition import analyze,summarize


class CompetitionTests(unittest.TestCase):
    def call(self,ids=('a','t','b'),pool=('t','a','b'),allowed=None,hashes=None,rank=None,scores=None):
        hashes=hashes or dict(a='A',t='T',b='B',r='R')
        d=dict(top20_ids=list(ids),top20_scores=scores or [.9-.1*i for i in range(len(ids))],target_id='t',baseline_rank=rank or ids.index('t')+1,original_query_row=1024)
        return analyze(d,list(pool),hashes,set(allowed or hashes),{'T'},'r')
    def test_positive_storage_not_rank(self):self.assertEqual(self.call()['visible_gt_rank'],2)
    def test_hidden_competitor(self):
        d=self.call(pool=('t','b'),allowed=('t','b','r'));self.assertTrue(d['hidden_competition']);self.assertEqual(d['visible_gt_rank'],1)
    def test_noncanonical_actual_gt_and_sha(self):
        d=self.call(pool=('t','b'),hashes=dict(a='T',t='T',b='B',r='R'));self.assertFalse(d['strongest_id_covered']);self.assertTrue(d['strongest_sha_covered']);self.assertFalse(d['strongest_negative_sha_covered'])
    def test_negative_alias(self):
        d=self.call(ids=('a','t','b','c'),hashes=dict(a='A',t='T',b='B',c='A',r='R'));self.assertEqual(d['candidates'][-1]['exclusion_reasons'],['retained_negative_byte_alias'])
    def test_empty(self):self.assertIsNone(summarize([])['full_margin']['mean']);self.assertIsNone(summarize([])['hidden_fraction_non_top1'])
    def test_duplicate_and_injection(self):
        for pool in (('t','a','a'),('t','z'),('a','t','b')):
            with self.assertRaises(ValueError):self.call(pool=pool)
    def test_source_conflict_and_nonfinite(self):
        with self.assertRaises(ValueError):self.call(rank=1)
        with self.assertRaises(ValueError):self.call(scores=[.9,float('nan'),.7])
    def test_top1_margin(self):
        d=self.call(ids=('t','a','b'));self.assertEqual(d['visible_gt_rank'],1);self.assertAlmostEqual(d['full_margin'],.1);self.assertEqual(d['full_margin'],d['visible_margin'])


if __name__=='__main__':unittest.main()
