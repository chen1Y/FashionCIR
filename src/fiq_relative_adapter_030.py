"""Frozen029 fusion plus4/8 projected relative residual parameters."""
import torch
from torch import nn
from torch.nn import functional as F
from fiq_fusion_029 import DQUFusionHead,TextMixtureGate,fuse_parts
from fiq_gate_patch_model_028 import AttributeGate

ARMS=('anchor_all','anchor_source','polarity_axis','target_minus_source')
class RelativeAdapter030(nn.Module):
    def __init__(self,state,arm):
        super().__init__()
        if arm not in ARMS:raise ValueError('Unknown030-P arm')
        self.arm=arm;self.attribute_gate=AttributeGate();self.text_mixture=TextMixtureGate();self.image_fusion=DQUFusionHead();self.attribute_gate.load_state_dict(state['attribute_gate']);self.text_mixture.load_state_dict(state['text_mixture']);self.image_fusion.load_state_dict(state['image_fusion'])
        self.attribute_gate.requires_grad_(False);self.text_mixture.requires_grad_(False);self.image_fusion.requires_grad_(False);self.blend=nn.Parameter(torch.zeros(4));self.gamma=nn.Parameter(torch.zeros(4),requires_grad=arm in ('polarity_axis','target_minus_source'));self.eval()
    def project(self):
        with torch.no_grad():self.blend.clamp_(0,.25);self.gamma.clamp_(0,.3)
    def forward(self,parts,reference,written,anchors,source,axis,active_all,active_source):
        mask=active_source if self.arm in ('anchor_source','target_minus_source') else active_all;mask=mask&parts['valid'];f=parts['slots'].float();b=self.blend[None,:,None]*mask[...,None];gamma=self.gamma[None,:,None]*mask[...,None];delta=axis if self.arm=='polarity_axis' else (f-source if self.arm=='target_minus_source' else torch.zeros_like(f));z=(1-b)*f+b*anchors+gamma*delta;normal=F.normalize(z,dim=-1)
        # Exact029 identity at zero, with the mathematically normalized derivative.
        zero=((b==0)&(gamma==0));modified=torch.where(zero,f+(normal-normal.detach()),normal);modified=torch.where(mask[...,None],modified,f);out=fuse_parts(self,dict(parts,slots=modified),reference,written,'full_learned_written');out['modified_slots']=modified;return out
