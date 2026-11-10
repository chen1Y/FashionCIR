from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fiq_small_math import adaptive_alpha, blend, block_rerank, ranks_from_scores


class SmallMathTests(unittest.TestCase):
    def test_endpoints_and_invalid_query_are_exact(self):
        left,right = torch.randn(3,8),torch.randn(3,8)
        self.assertTrue(torch.equal(blend(left,right,0),left))
        self.assertTrue(torch.equal(blend(left,right,1),right))
        alpha = adaptive_alpha(torch.tensor([0.,1.,.8]),torch.tensor([1,1,0]))
        torch.testing.assert_close(alpha,torch.tensor([.4,.6,0.]))
        self.assertTrue(torch.equal(blend(left,right,alpha)[2],left[2]))

    def test_topk_cannot_rescue_tail_and_weight_zero_is_baseline(self):
        scores = torch.tensor([[.9,.8,.7,.6]])
        attrs = torch.tensor([[-100.,100.,0.,1000.]])
        target = torch.tensor([3])
        self.assertEqual(block_rerank(scores,attrs,target,k=2).item(),4)
        self.assertTrue(torch.equal(block_rerank(scores,attrs,target,weight=0,k=2),ranks_from_scores(scores,target)))
        target = torch.tensor([1])
        self.assertEqual(block_rerank(scores,attrs,target,k=2).item(),1)

    def test_zero_attribute_preserves_ties(self):
        scores = torch.tensor([[.5,.5,.5,.1]])
        target = torch.tensor([1])
        self.assertTrue(torch.equal(block_rerank(scores,torch.zeros_like(scores),target,k=3),ranks_from_scores(scores,target)))


if __name__=='__main__':
    unittest.main()
