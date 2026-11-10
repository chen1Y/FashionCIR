from pathlib import Path
import importlib.util
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fiq_supported_head import select_supported_rows

TORCH=importlib.util.find_spec('torch') is not None
if TORCH:
    import copy
    import torch
    from fiq_candidate_head import CandidateHead,fit_statistics
    from fiq_supported_head import masked_loss


class SupportedSelectionTests(unittest.TestCase):
    def setUp(self):
        self.hashes={i:i for i in ('ref','gt','n1','n2','outside','alias')}
        self.hashes['alias']='gt'
        self.known={('ref','edit'):{'gt'}}
        self.allowed=set(self.hashes)

    def select(self,row):
        return select_supported_rows([row],self.hashes,self.allowed,self.known,['ref'],['edit'])

    def test_real_gt_already_in_head_and_alias_negative_excluded(self):
        row=dict(original_query_row=0,target_id='gt',structured_valid=True,
            top20_ids=['n1','alias','gt','n2'],inference_eligible_unique_negative_ids=['n1','n2'])
        selected,pools,why=self.select(row)
        self.assertEqual(selected,[0]);self.assertEqual(pools,[['gt','n1','n2']]);self.assertEqual(why,{'0':[]})

    def test_gt_not_in_head_is_masked_without_injection(self):
        row=dict(original_query_row=0,target_id='outside',structured_valid=True,
            top20_ids=['n1','n2'],inference_eligible_unique_negative_ids=['n1','n2'])
        selected,pools,why=self.select(row)
        self.assertFalse(selected);self.assertFalse(pools);self.assertIn('gt_outside_frozen_top20',why['0'])

    def test_invalid_query_and_no_negative_are_masked(self):
        row=dict(original_query_row=0,target_id='gt',structured_valid=False,
            top20_ids=['gt','alias'],inference_eligible_unique_negative_ids=[])
        selected,_,why=self.select(row)
        self.assertFalse(selected);self.assertEqual(why['0'],['invalid_structured','no_byte_eligible_head_negative'])

    def test_changed_candidate_evidence_is_rejected(self):
        row=dict(original_query_row=0,target_id='gt',structured_valid=True,
            top20_ids=['gt','n1'],inference_eligible_unique_negative_ids=['n2'])
        with self.assertRaises(ValueError):self.select(row)


@unittest.skipUnless(TORCH,'Torch checks run on experiment server')
class SupportedPaddingTests(unittest.TestCase):
    def pool(self):
        torch.manual_seed(42)
        return dict(x=torch.randn(2,20,6),base=torch.randn(2,20),valid=torch.ones(2,dtype=torch.bool),
                    mask=torch.tensor([[True]*3+[False]*17,[True]*5+[False]*15]))

    def test_padding_nan_does_not_change_loss_or_gradients(self):
        pool=self.pool();mean,std=fit_statistics(pool['x'][pool['mask']]);a=CandidateHead(mean,std);b=copy.deepcopy(a)
        other={k:v.clone() for k,v in pool.items()};other['x'][~other['mask']]=float('nan');other['base'][~other['mask']]=float('nan')
        first,second=masked_loss(a,pool),masked_loss(b,other)
        self.assertTrue(torch.equal(first,second));first.backward();second.backward()
        for p,q in zip(a.parameters(),b.parameters()):
            self.assertTrue(torch.equal(p.grad,q.grad));self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(a.network[-1].weight.grad.norm()),0)

    def test_padding_cannot_enter_fit_statistics(self):
        pool=self.pool();pool['x'][~pool['mask']]=float('nan');mean,std=fit_statistics(pool['x'][pool['mask']])
        self.assertTrue(torch.isfinite(mean).all());self.assertTrue(torch.isfinite(std).all())
        self.assertTrue((std>=1e-3).all())

    def test_missing_real_positive_or_negative_is_a_hard_failure(self):
        for count in (0,1):
            pool=self.pool();pool['mask'][0]=False;pool['mask'][0,:count]=True
            with self.assertRaises(ValueError):masked_loss(CandidateHead(torch.zeros(6),torch.ones(6)),pool)

    def test_zero_init_bounds_and_invalid_query_fallback_remain_unchanged(self):
        p=self.pool();h=CandidateHead(torch.zeros(6),torch.ones(6));self.assertTrue((h(p['x'],p['valid'])==0).all())
        h.network[-1].bias.data.fill_(100);p['valid'][1]=False;v=h(p['x'],p['valid'])
        self.assertLessEqual(float(v.detach().abs().max()),.03000001);self.assertTrue((v[1]==0).all())


if __name__=='__main__':unittest.main()
