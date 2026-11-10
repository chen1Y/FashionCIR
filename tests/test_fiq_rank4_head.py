from pathlib import Path
import importlib.util
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
TORCH=importlib.util.find_spec('torch') is not None
if TORCH:
    import copy
    import torch
    from fiq_candidate_head import CandidateHead
    from fiq_rank4_head import Rank4Head,pack_vectors
    from fiq_supported_head import masked_loss


@unittest.skipUnless(TORCH,'Torch checks run on experiment server')
class Rank4Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.scalar=CandidateHead(torch.zeros(6),torch.ones(6))
        self.scalar.network[-1].weight.data.normal_(0,.1)
        self.scalar.network[-1].bias.data.fill_(.1)
        self.x=torch.randn(3,20,6);self.q=torch.nn.functional.normalize(torch.randn(3,8),dim=-1)
        self.c=torch.nn.functional.normalize(torch.randn(3,20,8),dim=-1);self.valid=torch.tensor([True,False,True])
        self.mask=torch.tensor([[True]*4+[False]*16,[True]*3+[False]*17,[True]*8+[False]*12])

    def pool(self):
        return dict(x=pack_vectors(self.x,self.q,self.c,self.mask),mask=self.mask,valid=torch.ones(3,dtype=torch.bool),base=torch.randn(3,20))

    def test_both_arms_exactly_reproduce_nonzero_starting_scalar(self):
        packed=pack_vectors(self.x,self.q,self.c)
        for arm,count in [('scalar_continue',129),('rank4_branch',64)]:
            h=Rank4Head(copy.deepcopy(self.scalar),8,arm)
            self.assertTrue(torch.equal(h(packed,self.valid),self.scalar(self.x,self.valid)))
            self.assertEqual(sum(p.numel() for p in h.parameters() if p.requires_grad),count)

    def test_single_zero_projection_learns_and_query_projection_receives_gradient(self):
        h=Rank4Head(copy.deepcopy(self.scalar),8,'rank4_branch');p=self.pool()
        opt=torch.optim.AdamW([v for v in h.parameters() if v.requires_grad],lr=5e-4)
        masked_loss(h,p).backward();self.assertGreater(float(h.candidate_projection.weight.grad.norm()),0)
        opt.step();opt.zero_grad();masked_loss(h,p).backward()
        self.assertGreater(float(h.query_projection.weight.grad.norm()),0)

    def test_total_bound_is_single03_and_invalid_queries_stay_zero(self):
        h=Rank4Head(copy.deepcopy(self.scalar),8,'rank4_branch')
        h.query_projection.weight.data.fill_(100);h.candidate_projection.weight.data.fill_(100)
        out=h(pack_vectors(self.x,self.q,self.c),self.valid)
        self.assertLessEqual(float(out.detach().abs().max()),.03000001)
        self.assertTrue((out[1]==0).all())

    def test_padding_nan_cannot_change_loss_or_gradients_for_either_arm(self):
        for arm in ('scalar_continue','rank4_branch'):
            a=Rank4Head(copy.deepcopy(self.scalar),8,arm);b=copy.deepcopy(a);p=self.pool();bad={k:v.clone() for k,v in p.items()}
            bad['x'][~bad['mask']]=float('nan');bad['base'][~bad['mask']]=float('nan')
            loss1,loss2=masked_loss(a,p),masked_loss(b,bad);self.assertTrue(torch.equal(loss1,loss2));loss1.backward();loss2.backward()
            for pa,pb in zip(a.parameters(),b.parameters()):
                if pa.requires_grad:self.assertTrue(torch.equal(pa.grad,pb.grad));self.assertTrue(torch.isfinite(pa.grad).all())

    def test_branch_optimizer_cannot_change_frozen_scalar_or_normalization(self):
        h=Rank4Head(copy.deepcopy(self.scalar),8,'rank4_branch');before=copy.deepcopy(h.scalar.state_dict())
        opt=torch.optim.AdamW([v for v in h.parameters() if v.requires_grad],lr=5e-4);masked_loss(h,self.pool()).backward();opt.step()
        self.assertTrue(all(torch.equal(v,h.scalar.state_dict()[k]) for k,v in before.items()))

    def test_vector_branch_can_distinguish_candidates_with_equal_scalar_scores(self):
        scalar=CandidateHead(torch.zeros(6),torch.ones(6));h=Rank4Head(scalar,8,'rank4_branch')
        h.query_projection.weight.data.zero_();h.candidate_projection.weight.data.zero_()
        h.query_projection.weight.data[0,0]=1;h.candidate_projection.weight.data[0,1]=1
        q=torch.zeros(1,8);q[0,0]=1;c=torch.zeros(1,2,8);c[0,0,1]=1;c[0,1,1]=-1
        out=h(pack_vectors(torch.zeros(1,2,6),q,c),torch.ones(1,dtype=torch.bool))
        self.assertGreater(float(out[0,0].detach()),float(out[0,1].detach()))


if __name__=='__main__':unittest.main()
