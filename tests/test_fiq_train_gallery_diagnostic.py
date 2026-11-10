import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fiq_gallery_diagnostic_math import gallery_ids, rank_summary


class DiagnosticTests(unittest.TestCase):
    def test_train_gallery_excludes_every_heldout_and_deduplicates(self):
        ids,counts = gallery_ids(['s','t','t','otherclass_val','test'],{'otherclass_val','test'},
            [dict(candidate='s',target='t')])
        self.assertEqual(ids,['s','t'])
        self.assertEqual(counts['excluded_heldout_ids'],2)
        self.assertEqual(counts['raw_train_ids'],5)

    def test_missing_gold_is_not_injected_and_query_not_dropped(self):
        with self.assertRaises(ValueError):
            gallery_ids(['s','a'],set(),[dict(candidate='s',target='missing')])

    def test_heldout_query_and_identical_reference_are_rejected(self):
        with self.assertRaises(ValueError):gallery_ids(['s','t'],{'s'},[dict(candidate='s',target='t')])
        with self.assertRaises(ValueError):gallery_ids(['t'],set(),[dict(candidate='t',target='t')])

    def test_headroom_is_only_non_top1_targets_inside_the_head(self):
        result = rank_summary([1,2,20,21])
        self.assertEqual(result['recalls']['r1'],25)
        self.assertEqual(result['recalls']['r20'],75)
        self.assertEqual(result['top20_to_top1_oracle_headroom_pp'],50)
        self.assertEqual(result['outside_top20'],1)

    def test_saturated_baseline_has_no_top1_headroom(self):
        self.assertEqual(rank_summary([1]*8)['top20_to_top1_oracle_headroom_pp'],0)

    def test_missing_or_invalid_ranks_are_not_silently_filtered(self):
        for ranks in ([],[0],[True],[float('inf')]):
            with self.assertRaises(ValueError):rank_summary(ranks)


if __name__ == '__main__':
    unittest.main()
