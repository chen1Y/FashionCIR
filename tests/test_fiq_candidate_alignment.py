from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from fiq_candidate_alignment_math import analyze_row, summarize


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.ids = [f'i{i}' for i in range(60)]
        self.hashes = {i:i for i in self.ids}
        self.scores = [1.-i*.01 for i in range(20)]

    def run_row(self, rank, valid=True, scores=None, bound=.03):
        target = self.ids[rank-1]
        pool = [target]+[i for i in self.ids[20:] if i!=target][:31]
        return analyze_row(self.ids[:20], scores or self.scores, target, rank, pool, self.hashes,
                           set(self.ids), {target, 'i59'}, valid, bound)

    def test_outside_head_cannot_be_recovered_by_bounded_reranker(self):
        r = self.run_row(25)
        self.assertEqual((r['closed_bound_best_rank'],r['closed_bound_worst_rank']),(25,25))
        self.assertEqual(r['bucket'],'outside20')

    def test_invalid_structured_fallback_cannot_change_rank(self):
        r = self.run_row(5,False)
        self.assertEqual((r['closed_bound_best_rank'],r['closed_bound_worst_rank']),(5,5))

    def test_bound_improvement_and_regression_are_both_reported(self):
        self.assertTrue(self.run_row(4)['top1_newly_reachable'])
        self.assertTrue(self.run_row(1)['top1_can_be_lost'])
        self.assertTrue(self.run_row(12)['top10_newly_reachable'])

    def test_stable_tie_at_exact_bound_is_not_claimed_as_improvement(self):
        scores=[1.,.75]+[.5-i*.01 for i in range(18)]
        r=self.run_row(2,scores=scores,bound=.125)
        self.assertEqual(r['closed_bound_best_rank'],2)

    def test_sha_alias_pool_coverage_is_distinguished_from_id_coverage(self):
        self.hashes['i20']=self.hashes['i0']
        r=self.run_row(4)
        self.assertEqual(r['loss_negative_id_overlap_top20'],0)
        self.assertEqual(r['loss_negative_sha_overlap_top20'],1)

    def test_eligible_inference_negatives_exclude_positive_and_byte_aliases(self):
        self.hashes['i1']=self.hashes['i0']
        r=self.run_row(1)
        self.assertEqual(r['inference_eligible_unique_negative_count'],18)
        self.assertNotIn('i1',r['inference_eligible_unique_negative_ids'])

    def test_inconsistent_rank_and_nonfinite_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            analyze_row(self.ids[:20],self.scores,'i5',2,['i5']+self.ids[20:51],self.hashes,set(),set(),True)
        scores=self.scores.copy();scores[0]=float('nan')
        with self.assertRaises(ValueError):self.run_row(3,scores=scores)

    def test_summary_keeps_all_rows_and_reports_only_theoretical_ceiling(self):
        s=summarize([self.run_row(1),self.run_row(4),self.run_row(25)])
        self.assertEqual(s['queries'],3)
        self.assertEqual(s['forced_gt_outside_inference_top20'],1)
        self.assertAlmostEqual(s['closed_bound_optimistic_r1_upper'],200/3)


if __name__=='__main__':
    unittest.main()
