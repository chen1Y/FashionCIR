from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fiq_candidate_partition import image_bucket,validate_partitions

TORCH=importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from torch.nn import functional as F
    from fiq_candidate_head import (CandidateHead,candidate_features,fit_statistics,training_pool,shuffled_correction,head_ranks)
    from train_fiq_candidate_head import load_val_after_fit


class PartitionTests(unittest.TestCase):
    def setUp(self):
        ids=[f'image{i}' for i in range(300)]
        self.fit=[x for x in ids if image_bucket(x)!=0];self.dev=[x for x in ids if image_bucket(x)==0]
        self.ids=ids

    def test_negative_banks_exclude_the_entire_other_partition(self):
        fit,dev=validate_partitions([dict(candidate=self.fit[0],target=self.fit[1])],
            [dict(candidate=self.dev[0],target=self.dev[1])],self.ids,set())
        self.assertFalse(set(fit)&set(dev))
        self.assertTrue(all(image_bucket(self.ids[i])!=0 for i in fit))
        self.assertTrue(all(image_bucket(self.ids[i])==0 for i in dev))

    def test_heldout_gallery_or_cross_partition_query_rejected(self):
        for heldout,fit in [({self.ids[0]},[]),(set(),[dict(candidate=self.fit[0],target=self.dev[0])])]:
            with self.assertRaises(ValueError):validate_partitions(fit,[],self.ids,heldout)

    def test_duplicate_or_missing_positive_cannot_be_silently_repaired(self):
        with self.assertRaises(ValueError):validate_partitions([],[],self.ids+[self.ids[0]],set())
        with self.assertRaises(ValueError):validate_partitions([dict(candidate=self.fit[0],target='missing')],[],self.ids,set())


@unittest.skipUnless(TORCH,'Torch-dependent checks run on the experiment server')
class CandidateHeadTests(unittest.TestCase):
    def test_zero_initialization_bounds_and_invalid_fallback(self):
        head=CandidateHead(torch.zeros(6),torch.ones(6));x=torch.randn(3,20,6);valid=torch.tensor([True,False,True])
        self.assertTrue(torch.equal(head(x,valid),torch.zeros(3,20)))
        head.network[-1].bias.data.fill_(100)
        out=head(x,valid);self.assertLessEqual(float(out.abs().max()),.03000001)
        self.assertTrue(torch.equal(out[1],torch.zeros(20)))

    def test_fit_statistics_are_not_recomputed_from_dev(self):
        x=torch.randn(8,32,6);mean,std=fit_statistics(x);head=CandidateHead(mean,std)
        head(torch.ones(3,20,6)*100,torch.ones(3,dtype=torch.bool))
        self.assertTrue(torch.equal(head.mean,mean));self.assertTrue(torch.equal(head.std,std))
        self.assertTrue((std>=1e-3).all())

    def test_pool_excludes_reference_and_known_alternative_positive(self):
        gallery=F.normalize(torch.randn(50,8),dim=-1);qd=gallery[:2];qq=qd.clone()
        rows=[dict(candidate='i0',target='i2',modification_text='red'),dict(candidate='i0',target='i3',modification_text='red')]
        pool,_=training_pool(qd,qq,gallery,torch.tensor([0,0]),torch.tensor([2,3]),torch.ones(50,dtype=torch.bool),rows,[f'i{i}' for i in range(50)])
        self.assertFalse(torch.isin(pool[:,1:],torch.tensor([0,2,3])).any())
        self.assertTrue(torch.equal(pool[:,0],torch.tensor([2,3])))

    def test_head_only_ranking_and_shuffle_preserve_tail(self):
        order=torch.arange(30).repeat(2,1);targets=torch.tensor([0,25]);scores=torch.arange(20).float().repeat(2,1)
        rank=head_ranks(order,targets,scores)
        self.assertEqual(rank.tolist(),[20,26])
        shuffled=shuffled_correction(scores)
        self.assertTrue(torch.equal(shuffled.sort(-1).values,scores.sort(-1).values))

    def test_training_updates_parameters_with_finite_gradients(self):
        torch.manual_seed(42);x=torch.randn(16,32,6);mean,std=fit_statistics(x);head=CandidateHead(mean,std)
        initial=head.network[-1].weight.detach().clone();optimizer=torch.optim.AdamW(head.parameters(),lr=5e-4)
        for _ in range(5):
            optimizer.zero_grad();value=F.cross_entropy(head(x,torch.ones(16,dtype=torch.bool))/.07,torch.zeros(16,dtype=torch.long));value.backward()
            self.assertTrue(torch.isfinite(value));self.assertTrue(torch.isfinite(torch.nn.utils.clip_grad_norm_(head.parameters(),1.)))
            optimizer.step()
        self.assertGreater(float((head.network[-1].weight-initial).norm()),0)

    def test_official_features_cannot_load_before_fit_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('train_fiq_candidate_head.torch.load') as loader:
                with self.assertRaises(ValueError):load_val_after_fit(SimpleNamespace(output=Path(folder)),{})
                loader.assert_not_called()


if __name__=='__main__':unittest.main()
