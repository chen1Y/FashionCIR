import sys,unittest
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parent)]
try:
 import torch
 from torch import nn
 from attr_token_pair_alignment import HookedVocabulary,PairAlignedText,paired_losses
 from test_attr_modules import Clip,Tokenizer
except ImportError:torch=None

@unittest.skipIf(torch is None,'Torch required on server')
class PairTests(unittest.TestCase):
 def test_hook_and_adamw_preserve_old_rows(self):
  e=HookedVocabulary(nn.Embedding(64,32));old=e.weight[:64].detach().clone();start=e.weight[64:].detach().clone()
  optimizer=torch.optim.AdamW([dict(params=[e.weight],lr=.01,weight_decay=0.)])
  for _ in range(3):
   optimizer.zero_grad();e(torch.arange(68)).square().sum().backward()
   self.assertEqual(float(e.weight.grad[:64].abs().sum()),0.);self.assertGreater(float(e.weight.grad[64:].norm()),0.);optimizer.step()
  self.assertTrue(torch.equal(old,e.weight[:64]));self.assertFalse(torch.equal(start,e.weight[64:]))
 def test_only_last_two_blocks(self):
  c=Clip();c.transformer.resblocks=nn.ModuleList([type(c.transformer.resblocks[0])(32) for _ in range(4)])
  m=PairAlignedText(c,Tokenizer(),2)
  self.assertTrue(all(not p.requires_grad for p in c.transformer.resblocks[0].parameters()))
  self.assertTrue(all(not p.requires_grad for p in c.transformer.resblocks[1].parameters()))
  self.assertTrue(all(p.requires_grad for p in c.transformer.resblocks[2:].parameters()))
  self.assertFalse(c.token_embedding.weight.requires_grad);self.assertFalse(c.text_projection.requires_grad)
 def test_same_initial_output_and_delta_roundtrip(self):
  import copy
  c=Clip();a=PairAlignedText(copy.deepcopy(c),Tokenizer(),0);b=PairAlignedText(copy.deepcopy(c),Tokenizer(),2)
  rows=[dict(color='navy',shape='long',detail='collar',texture='cotton')]*2
  self.assertTrue(torch.equal(a(rows)['f_attr_text'],b(rows)['f_attr_text']))
  s=a.delta_state();a.embedding.weight.data[-4:]+=1.;a.load_delta(s)
  self.assertTrue(torch.equal(a.embedding.weight[-4:],s['new_token_rows']))
 def test_invalid_slot_nan_teacher_mask(self):
  m=PairAlignedText(Clip(),Tokenizer(),0);out=m([dict(color='navy'),dict(shape='long')]);v=torch.nn.functional.normalize(torch.randn(2,32),dim=-1)
  t=torch.nn.functional.normalize(torch.randn(2,3,32),dim=-1);t[:,2]=float('nan');mask=torch.tensor([[True,True,False]]*2)
  teacher=torch.randn(2,4,32);teacher[~out['attr_valid']]=float('nan')
  losses=paired_losses(out,v,v,t,mask,teacher);losses['total'].backward()
  self.assertTrue(torch.isfinite(losses['total']));self.assertEqual(float(m.embedding.weight.grad[:64].abs().sum()),0.)

if __name__=='__main__':unittest.main()
