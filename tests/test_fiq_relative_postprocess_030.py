import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fiq_relative_postprocess_030 import analyse,derive,source_phrase
try:import torch
except ImportError:torch=None
class RelativePost030Tests(unittest.TestCase):
    def setUp(self):self.base=dict(color='lighter than reference',shape='crew neck',detail='graphic',texture='solid')
    def test_black_lighter_three_separate_retained_cues(self):
        d=derive('lighter color',self.base,'a black shirt','ok',str.split);self.assertEqual(len(d['hypotheses']),3);self.assertEqual(d['hypotheses'][0]['color'],'charcoal, lighter than reference');self.assertTrue(all(h['detail']=='graphic' for h in d['hypotheses']))
    def test_explicit_target_and_wrong_source_not_enumerated(self):
        self.assertEqual(derive('lighter white color',self.base,'a black shirt','ok',str.split)['hypotheses'],[]);self.assertEqual(source_phrase('a black and white shirt','color','lightness'),'')
    def test_comparative_not_absolute_and_region(self):
        self.assertEqual(analyse('long sleeves'),[]);self.assertEqual(analyse('longer sleeves')[0]['axis'],'sleeve_length');self.assertEqual(analyse('longer hem')[0]['axis'],'garment_length')
    def test_old_unknown_and_conflicting_axes_fallback(self):
        self.assertEqual(derive('lighter color',self.base,'a black shirt','fallback',str.split)['hypotheses'],[]);self.assertEqual(analyse('lighter and darker color'),[])
    def test_no_failed_online_payload_dependency(self):
        self.assertEqual(derive('lighter color',self.base,'a shirt','ok',str.split)['hypotheses'],[])
@unittest.skipIf(torch is None,'Server Torch required')
class RelativeAdapter030Tests(unittest.TestCase):
    def test_exact_identity_gradient_and_projection(self):
        from fiq_relative_adapter_030 import RelativeAdapter030
        from fiq_fusion_029 import TextMixtureGate,DQUFusionHead,fuse_parts
        from fiq_gate_patch_model_028 import AttributeGate
        state=dict(attribute_gate=AttributeGate().state_dict(),text_mixture=TextMixtureGate().state_dict(),image_fusion=DQUFusionHead().state_dict());model=RelativeAdapter030(state,'polarity_axis');norm=lambda x:torch.nn.functional.normalize(x,dim=-1);parts=dict(mod=norm(torch.randn(2,1024)),eos=norm(torch.randn(2,1024)),slots=norm(torch.randn(2,4,1024)),valid=torch.ones(2,4,dtype=torch.bool));ref=norm(torch.randn(2,1024));written=norm(torch.randn(2,1024));anchor=norm(torch.randn(2,4,1024));axis=norm(torch.randn(2,4,1024));mask=torch.ones(2,4,dtype=torch.bool);expected=fuse_parts(model,parts,ref,written,'full_learned_written');actual=model(parts,ref,written,anchor,anchor,axis,mask,mask);self.assertTrue(torch.equal(expected['q'],actual['q']));actual['q'][:,0].sum().backward();self.assertGreater(float(model.gamma.grad.norm()),0.);self.assertFalse(any(p.grad is not None for p in model.image_fusion.parameters()))
        with torch.no_grad():model.blend[:]=1;model.gamma[:]=-1
        model.project();self.assertTrue(torch.equal(model.blend,torch.full((4,),.25)));self.assertEqual(int(model.gamma.count_nonzero()),0)
if __name__=='__main__':unittest.main()
