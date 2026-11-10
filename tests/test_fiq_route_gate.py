from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fiq_route_gate import sample_partitions,features,RouteGate,mixed_query,contrastive_loss,shuffled_valid_alpha


class GateTests(unittest.TestCase):
    def test_split_is_disjoint_and_deterministic(self):
        rows=[dict(candidate=f'i{i}',target=f'j{i}',source_index=i) for i in range(300)]
        fit,dev,counts=sample_partitions(rows,{'i2','j3'},train_limit=100,dev_limit=20)
        ids=lambda rows:{r[k] for r in rows for k in ('candidate','target')}
        self.assertFalse(ids(fit)&ids(dev));self.assertFalse((ids(fit)|ids(dev))&{'i2','j3'})
        self.assertEqual((fit,dev,counts),sample_partitions(list(reversed(rows)),{'i2','j3'},train_limit=100,dev_limit=20))

    def test_gate_initialization_bounds_and_invalid_fallback(self):
        gate=RouteGate();valid=torch.tensor([True,False,True]);alpha=gate(torch.randn(3,4),valid)
        torch.testing.assert_close(alpha,torch.tensor([.5,0,.5]))
        gate.network[-1].bias.data.fill_(100)
        alpha=gate(torch.randn(3,4),valid)
        self.assertLessEqual(float(alpha.max()),.75);self.assertEqual(float(alpha[1]),0)
        qd,qq=F.normalize(torch.randn(3,8),dim=-1),F.normalize(torch.randn(3,8),dim=-1)
        self.assertTrue(torch.equal(mixed_query(qd,qq,alpha)[1],qd[1]))

    def test_exact_repeated_targets_are_positives(self):
        q=F.normalize(torch.randn(2,8),dim=-1)
        self.assertAlmostEqual(float(contrastive_loss(q,q,['s','s'],['same','same'])),0,places=6)

    def test_shuffling_preserves_validity_and_weight_distribution(self):
        alpha=torch.tensor([.3,0.,.6,.7,0.]);valid=alpha>0
        shuffled=shuffled_valid_alpha(alpha,valid)
        self.assertTrue(torch.equal(shuffled[~valid],alpha[~valid]))
        self.assertTrue(torch.equal(shuffled[valid].sort().values,alpha[valid].sort().values))

    def test_supervised_train_loss_updates_gate_finitely(self):
        torch.manual_seed(42);qd=F.normalize(torch.randn(8,16),dim=-1);qq=F.normalize(torch.randn(8,16),dim=-1)
        gate=RouteGate();x=features(qd,qq,torch.ones(8),torch.ones(8,dtype=torch.bool));opt=torch.optim.Adam(gate.parameters(),lr=.01)
        before=gate.network[-1].weight.detach().clone()
        for _ in range(5):
            opt.zero_grad();alpha=gate(x,torch.ones(8,dtype=torch.bool));loss=contrastive_loss(mixed_query(qd,qq,alpha),qd,[f's{i}' for i in range(8)],[f't{i}' for i in range(8)]);loss.backward();opt.step();self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float((gate.network[-1].weight-before).norm()),0)


if __name__=='__main__':
    unittest.main()
