import sys,unittest,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from expanded_attr_schema import validate,request_key
try:
 import torch
 from expanded_attr_model import legal_masks,ranks_from_scores,metrics,selection_key,compose
except ImportError:torch=None

class RelativeTests(unittest.TestCase):
 def row(self,shape):return dict(target_attrs=dict(color='brown',shape=shape,detail='',texture='parrot pattern'),uncertainties=[])
 def test_absolute_long_is_not_longer(self):self.assertIn('missing_comparative_longer',validate(self.row('long sleeves'),'longer sleeves',str.split))
 def test_relative_preserved(self):self.assertEqual(validate(self.row('sleeves longer than reference'),'longer sleeves',str.split),[])
 def test_new_prompt_identity(self):self.assertNotEqual(request_key('a','shorter'),request_key('a','longer'))

@unittest.skipIf(torch is None,'Server Torch required')
class ExpandedMathTests(unittest.TestCase):
 def test_actual_gt_noncanonical_and_unique_negatives(self):
  ids=['ref','canonical_gt','actual_gt','neg','neg_alias','known_other'];hashes=dict(ref='r',canonical_gt='g',actual_gt='g',neg='n',neg_alias='n',known_other='k')
  r=dict(candidate='ref',target='actual_gt',modification_text='edit')
  m=dict(banks=dict(fit=ids),canonical=dict(fit=['ref','canonical_gt','neg','known_other']),image_sha256=hashes,offsets=dict(fit=[0]),records=[r],known_positive_sha={json.dumps(('r','edit')):['g','k']})
  masks,labels=legal_masks(m,'fit');self.assertEqual(labels.tolist(),[2]);self.assertEqual(masks[0].nonzero().flatten().tolist(),[2,3])
 def test_reference_exclusion_and_stable_id_tie(self):
  s=torch.tensor([[.9,.5,.5,.4]]);r=ranks_from_scores(s,torch.tensor([2]),torch.tensor([0]));self.assertEqual(r.tolist(),[2])
 def test_r10_r50_selection_dominates_r1_and_nce(self):
  a=dict(r1=90.,r10=60.,r50=80.);b=dict(r1=10.,r10=65.,r50=80.)
  self.assertGreater(selection_key(b,10.),selection_key(a,1.));self.assertGreater(selection_key(a,1.),selection_key(a,2.))
 def test_full_gallery_metrics_include_r50(self):
  d=metrics([1,10,11,50,51]);self.assertEqual((d['r1'],d['r10'],d['r50']),(20.,40.,80.))
 def test_dual_keeps_independent_original_signal(self):
  ref=torch.tensor([[0.,0.,1.]]);a=dict(text=torch.tensor([[1.,0.,0.]]));b=dict(text=torch.tensor([[.5,.5,0.]]))
  self.assertFalse(torch.equal(compose(a,ref,'dual'),compose(b,ref,'dual')))

if __name__=='__main__':unittest.main()
