from pathlib import Path
import importlib.util
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fiq_rank4_control_math import transition_counts,branch_controls
TORCH=importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from fiq_candidate_head import shuffled_correction


class TransitionTests(unittest.TestCase):
    def test_all_queries_accounted_and_net_pp_matches_rk_change(self):
        r=transition_counts([1,5,12,30],[2,1,8,31],1)
        self.assertEqual((r['gains'],r['losses'],r['both_hit'],r['neither_hit']),(1,1,0,2));self.assertEqual(r['delta_pp'],0)
        r=transition_counts([1,5,12,30],[2,1,8,31],10);self.assertEqual(r['net_queries'],1);self.assertEqual(r['delta_pp'],25)

    def test_inconsistent_or_invalid_ranks_fail(self):
        for a,b in [([],[]),([1],[1,2]),([0],[1])]:
            with self.assertRaises(ValueError):transition_counts(a,b,1)


@unittest.skipUnless(TORCH,'Torch checks run on experiment server')
class BranchControlTests(unittest.TestCase):
    def test_only_lowrank_is_shuffled_and_frozen_logit_stays_aligned(self):
        logit=torch.arange(40,dtype=torch.float32).reshape(2,20)/20;u=torch.arange(40,dtype=torch.float32).reshape(2,20)/100
        c=branch_controls(logit,u,torch.ones(2,dtype=torch.bool))
        self.assertTrue(torch.equal(c['rank4_only_shuffle'],.03*torch.tanh(logit+shuffled_correction(u))))
        self.assertTrue(torch.equal(c['whole_correction_shuffle'],shuffled_correction(c['paired_rank4'])))
        self.assertFalse(torch.equal(c['rank4_only_shuffle'],c['whole_correction_shuffle']))

    def test_zero_rank4_preserves_original_while_whole_shuffle_need_not(self):
        logit=torch.arange(20,dtype=torch.float32)[None,:]/10;c=branch_controls(logit,torch.zeros_like(logit),torch.tensor([True]))
        self.assertTrue(torch.equal(c['zero_rank4'],c['rank4_only_shuffle']));self.assertTrue(torch.equal(c['zero_rank4'],c['paired_rank4']))
        self.assertFalse(torch.equal(c['zero_rank4'],c['whole_correction_shuffle']))

    def test_invalid_fallback_and_single_total_bound(self):
        logit=torch.ones(2,20)*100;u=-torch.ones_like(logit)*10;c=branch_controls(logit,u,torch.tensor([True,False]))
        for correction in c.values():
            self.assertLessEqual(float(correction.abs().max()),.03000001);self.assertTrue((correction[1]==0).all())

    def test_nan_padding_never_moves_into_real_candidates(self):
        logit=torch.tensor([[1.,2.,float('nan'),float('nan')]]);u=logit.clone();mask=torch.tensor([[True,True,False,False]])
        c=branch_controls(logit,u,torch.tensor([True]),mask)
        for correction in c.values():self.assertTrue(torch.isfinite(correction).all());self.assertTrue((correction[~mask]==0).all())


if __name__=='__main__':unittest.main()
