import sys,unittest
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parent)]
try:
 import torch
 from attr_token_pair_alignment import PairAlignedText,paired_losses
 from train_clean_attr_alignment import plain_text,loss_for,token_stats
 from test_attr_modules import Clip,Tokenizer
except ImportError:torch=None

@unittest.skipIf(torch is None,'Torch required on server')
class CleanTrainingTests(unittest.TestCase):
 def test_unchanged_loss_when_all_slots_valid(self):
  m=PairAlignedText(Clip(),Tokenizer(),2);o=m([dict(color='white',shape='loose',detail='collar',texture='dots')]*2)
  v=torch.nn.functional.normalize(torch.randn(2,32),dim=-1);t=torch.nn.functional.normalize(torch.randn(2,3,32),dim=-1);mask=torch.ones(2,3,dtype=torch.bool);teacher=torch.randn(2,4,32)
  a=paired_losses(o,v,v,t,mask,teacher);b=loss_for(o,v,v,t,mask,teacher)
  for k in a:self.assertTrue(torch.allclose(a[k],b[k]))
 def test_empty_slot_global_loss_zero(self):
  m=PairAlignedText(Clip(),Tokenizer(),2);o=m([{}]*2);v=torch.nn.functional.normalize(torch.randn(2,32),dim=-1);t=torch.nn.functional.normalize(torch.randn(2,3,32),dim=-1)
  d=loss_for(o,v,v,t,torch.ones(2,3,dtype=torch.bool),torch.zeros(2,4,32));self.assertEqual(float(d['global_alignment']),0.);self.assertTrue(torch.isfinite(d['total']))
 def test_interpretable_token_statistics(self):
  m=PairAlignedText(Clip(),Tokenizer(),2);initial=m.delta_state();stats=token_stats(m,initial)
  self.assertEqual(stats['color']['delta_norm'],0.);self.assertAlmostEqual(stats['color']['cosine_to_initial'],1.,places=5)

if __name__=='__main__':unittest.main()
