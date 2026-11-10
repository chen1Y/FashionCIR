import unittest,sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fiq_gate_patch_math_028 import masked_weights,summax_reference
try:import torch
except ImportError:torch=None


class GatePatchScalarTests(unittest.TestCase):
    def test_uniform_and_missing_slots(self):
        self.assertEqual(masked_weights([0.,0.,0.,0.],[True,False,True,False]),[.5,0.,.5,0.]);self.assertEqual(masked_weights([0.]*4,[False]*4),[0.]*4)
        self.assertAlmostEqual(sum(masked_weights([100.,-100.],[True,True])),1.)
    def test_sum_of_max_is_not_max_of_sum(self):
        self.assertEqual(summax_reference([[1.,0.],[0.,1.]],[True,True],[[1.,0.],[0.,1.]]),2.)
        self.assertEqual(summax_reference([[1.,0.],[0.,1.]],[True,False],[[1.,0.],[0.,1.]]),1.)
        self.assertIsNone(summax_reference([[1.,0.]],[False],[[1.,0.]]))
    def test_bad_scalar_inputs_rejected(self):
        with self.assertRaises(ValueError):masked_weights([1.],[True,False])
        with self.assertRaises(ValueError):summax_reference([[0.,0.]],[True],[[1.,0.]])


@unittest.skipIf(torch is None,'Server Torch required')
class GatePatchTensorTests(unittest.TestCase):
    def test_gate_initialization_exactly_uniform_on_valid_slots(self):
        from fiq_gate_patch_model_028 import AttributeGate
        gate=AttributeGate();slots=torch.nn.functional.normalize(torch.randn(3,4,5),dim=-1);e=torch.randn(3,5);v=torch.randn(3,5);valid=torch.tensor([[1,1,1,1],[1,0,1,0],[0,0,0,0]],dtype=torch.bool);weights=gate(slots,e,v,valid)
        self.assertTrue(torch.isfinite(weights).all());self.assertTrue(torch.equal(weights[0],torch.full((4,),.25)));self.assertTrue(torch.equal(weights[1],torch.tensor([.5,0.,.5,0.])));self.assertEqual(int(weights[2].count_nonzero()),0);self.assertEqual(sum(p.numel() for p in gate.parameters()),145)
    def test_vectorized_local_cosine_matches_independent_scalar_reference(self):
        from fiq_gate_patch_model_028 import local_scores
        torch.manual_seed(42);s=torch.randn(2,4,5);p=torch.randn(3,7,5);valid=torch.tensor([[1,1,1,1],[1,0,1,0]],dtype=torch.bool);score,_,winner=local_scores(s,valid,p)
        for b in range(2):
            for c in range(3):self.assertAlmostEqual(float(score[b,c]),summax_reference(s[b].tolist(),valid[b].tolist(),p[c].tolist()),places=5)
        self.assertEqual(tuple(winner.shape),(2,4,3))
    def test_gate_can_receive_nonzero_gradient(self):
        from fiq_gate_patch_model_028 import AttributeGate
        gate=AttributeGate();s=torch.nn.functional.normalize(torch.randn(2,4,5),dim=-1);weights=gate(s,torch.randn(2,5),torch.randn(2,5),torch.ones(2,4,dtype=torch.bool));(weights*torch.tensor([1.,2.,3.,4.])).sum().backward();self.assertGreater(float(gate.mlp[-1].weight.grad.norm()),0.)


if __name__=='__main__':unittest.main()
