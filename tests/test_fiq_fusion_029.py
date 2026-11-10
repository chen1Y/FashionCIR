import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
try:import torch
except ImportError:torch=None

@unittest.skipIf(torch is None,'Torch server required')
class Fusion029Tests(unittest.TestCase):
    def test_mix_uniform_bounded_and_invalid(self):
        from fiq_fusion_029 import TextMixtureGate
        gate=TextMixtureGate();a=torch.nn.functional.normalize(torch.randn(3,8),dim=-1);valid=torch.tensor([[1,1,1,1],[1,0,0,0],[0,0,0,0]],dtype=torch.bool);w=gate(a,a,a,valid);self.assertTrue(torch.equal(w,torch.tensor([.5,.5,0.])));w.sum().backward();self.assertGreater(float(gate.network[-1].weight.grad.norm()),0.)
    def test_image_fusion_normalized_and_gradients(self):
        from fiq_fusion_029 import DQUFusionHead
        model=DQUFusionHead(8,0.);x=torch.randn(3,8,requires_grad=True);v=torch.randn(3,8);q,a=model(x,v);self.assertTrue(torch.allclose(q.norm(dim=-1),torch.ones(3)));self.assertTrue(((a>0)&(a<1)).all());q[:,0].sum().backward();self.assertGreater(float(x.grad.norm()),0.);self.assertGreater(float(model.combiner_fc[0].weight.grad.norm()),0.)
    def test_equal_gate_and_full_initial_match_mean(self):
        from fiq_fusion_029 import TextMixtureGate,DQUFusionHead,fuse_parts
        from fiq_gate_patch_model_028 import AttributeGate
        from types import SimpleNamespace
        model=SimpleNamespace(attribute_gate=AttributeGate(),text_mixture=TextMixtureGate(),image_fusion=DQUFusionHead(8,0.));normal=lambda x:torch.nn.functional.normalize(x,dim=-1);valid=torch.tensor([[1,1,1,1],[1,0,1,0],[0,0,0,0]],dtype=torch.bool);p=dict(mod=normal(torch.randn(3,8)),eos=normal(torch.randn(3,8)),slots=normal(torch.randn(3,4,8)),valid=valid);v=normal(torch.randn(3,8));written=normal(torch.randn(3,8));x=fuse_parts(model,p,v,written,'mean_dual_written')
        for arm in ('gated_dual_written','full_learned_written'):
            y=fuse_parts(model,p,v,written,arm);self.assertTrue(torch.allclose(x['q'],y['q'],atol=1e-6));self.assertEqual(float(y['beta'][-1]),0.)
    def test_native_dqu_head_reproduction(self):
        from fiq_fusion_029 import DQUFusionHead
        from torch.nn import functional as F
        m=DQUFusionHead(8,0.);t=F.normalize(torch.randn(2,8),dim=-1);v=F.normalize(torch.randn(2,8),dim=-1);a=m.scaler_fc(m.combiner_fc(torch.cat([t,v],-1)));expected=F.normalize(a*t+(1-a)*v,dim=-1);actual,_=m(t,v);self.assertTrue(torch.allclose(actual,expected,atol=1e-6))

if __name__=='__main__':unittest.main()
