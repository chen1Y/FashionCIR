from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
try:
    import torch
    from attr_slot_rank_math import candidate_ranks
except ImportError:torch=None


@unittest.skipIf(torch is None,'Server Torch suite required')
class CandidateRankTests(unittest.TestCase):
    def test_tie_uses_original_order_not_positive_column(self):
        r=candidate_ranks(torch.tensor([[.5,.5,.4]]),torch.ones(1,3,dtype=torch.bool),torch.tensor([[5,1,9]]));self.assertEqual(r.tolist(),[2])
    def test_mask_and_ranks(self):
        r=candidate_ranks(torch.tensor([[.3,.2,float('nan')],[.2,.3,.4]]),torch.tensor([[1,1,0],[1,1,1]],dtype=torch.bool),torch.tensor([[2,1,99],[1,2,3]]));self.assertEqual(r.tolist(),[1,3])
    def test_nonfinite_real_rejected(self):
        with self.assertRaises(FloatingPointError):candidate_ranks(torch.tensor([[float('nan'),.2]]),torch.ones(1,2,dtype=torch.bool),torch.tensor([[1,2]]))


if __name__=='__main__':unittest.main()
