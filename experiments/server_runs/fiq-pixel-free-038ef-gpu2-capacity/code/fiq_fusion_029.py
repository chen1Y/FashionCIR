"""029 independent original-text / gated-attribute / written-image fusion."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from fiq_gate_patch_model_028 import GatedAttributeText
from expanded_attr_model import branch_forward as old_branch

ARMS=('original_written','mean_dual_written','gated_dual_written','full_learned_written')


class TextMixtureGate(nn.Module):
    def __init__(self):
        super().__init__();self.network=nn.Sequential(nn.Linear(5,16),nn.GELU(),nn.Linear(16,1));nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
    def forward(self,mod,generated,reference,valid):
        x=torch.stack([(mod*generated).sum(-1),(mod*reference).sum(-1),(generated*reference).sum(-1),valid.float().mean(-1),valid.any(-1).float()],-1)
        return (.25+.5*self.network(x).float().sigmoid()).squeeze(-1)*valid.any(-1)


class DQUFusionHead(nn.Module):
    def __init__(self,dimension=1024,dropout=.5):
        super().__init__();self.combiner_fc=nn.Sequential(nn.Linear(2*dimension,dimension),nn.ReLU());self.dropout=nn.Dropout(dropout);self.scaler_fc=nn.Sequential(nn.Linear(dimension,dimension),nn.ReLU(),nn.Dropout(dropout),nn.Linear(dimension,1),nn.Sigmoid())
    def forward(self,text,written):
        text=F.normalize(text.float(),dim=-1);written=F.normalize(written.float(),dim=-1);a=self.scaler_fc(self.dropout(self.combiner_fc(torch.cat([text,written],-1)))).float();return F.normalize(a*text+(1-a)*written,dim=-1),a.squeeze(-1)
    def from_dqu(self,path):
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)['model_state'];self.combiner_fc.load_state_dict({k[len('combiner_fc.'):]:v for k,v in state.items() if k.startswith('combiner_fc.')});self.scaler_fc.load_state_dict({k[len('scaler_fc.'):]:v for k,v in state.items() if k.startswith('scaler_fc.')})


class FusionText029(GatedAttributeText):
    def __init__(self,clip,tokenizer,last_blocks=2):
        super().__init__(clip,tokenizer,last_blocks);self.text_mixture=TextMixtureGate();self.image_fusion=DQUFusionHead();self.active_arm='full_learned_written'
    def train(self,mode=True):
        super().train(mode);self.clip.eval();return self
    def parameter_groups(self):
        groups=super().parameter_groups()+[dict(params=list(self.text_mixture.parameters()),lr=2e-4,weight_decay=.01),dict(params=list(self.image_fusion.parameters()),lr=1e-4,weight_decay=.01)]
        return [dict(g,params=[p for p in g['params'] if p.requires_grad]) for g in groups if any(p.requires_grad for p in g['params'])]
    def configure_arm(self,arm):
        if arm not in ARMS:raise ValueError('Unregistered029 arm')
        self.active_arm=arm;self.attribute_gate.requires_grad_(arm in ('gated_dual_written','full_learned_written'));self.text_mixture.requires_grad_(arm=='full_learned_written')
    def delta_state(self):
        out=super().delta_state()
        for name in ('text_mixture','image_fusion'):out[name]={k:v.detach().cpu().clone() for k,v in getattr(self,name).state_dict().items()}
        return out
    def load_delta(self,state):
        super().load_delta(state)
        for name in ('text_mixture','image_fusion'):
            if name in state:getattr(self,name).load_state_dict(state[name])


def encode_parts(model,rows,arm):
    if arm=='original_written':
        o=old_branch(model,rows,'original');return dict(mod=o['t_mod'],eos=o['global_text'],slots=None,valid=None)
    out=old_branch(model,rows,'dual');return dict(mod=out['t_mod'],eos=out['global_text'],slots=out['slots'],valid=out['valid'])


def fuse_text(model,parts,reference,arm):
    mod=parts['mod'].float();eos=parts['eos'].float();valid=parts['valid'];weights=None;beta=torch.zeros(len(mod),device=mod.device)
    if arm=='original_written':text=mod
    else:
        slots=parts['slots'].float()
        if arm=='mean_dual_written':weights=valid.float()/valid.sum(-1,keepdim=True).clamp_min(1)
        else:weights=model.attribute_gate(slots,eos,reference,valid)
        generated=F.normalize(eos+(slots*weights[...,None]).sum(1),dim=-1)
        beta=model.text_mixture(mod,generated,reference,valid) if arm=='full_learned_written' else .5*valid.any(-1).float()
        text=F.normalize((1-beta[:,None])*mod+beta[:,None]*generated,dim=-1)
    return text,beta,weights


def fuse_parts(model,parts,reference,written,arm):
    text,beta,weights=fuse_text(model,parts,reference,arm)
    q,alpha=model.image_fusion(text,written)
    return dict(q=q,text=text,alpha=alpha,beta=beta,slot_weights=weights,global_text=parts['eos'],slots=parts['slots'],valid=parts['valid'])


def objective(out,bank,labels,mask,teacher,arm,*,reference=None,reference_weight=0.):
    score=out['q']@bank.T;logits=score.masked_fill(~mask,-torch.inf)/.07;target=bank[labels]
    if reference_weight:
        ref_logit=(out['q'].float()*reference.float()).sum(-1)/.07
        den=torch.logaddexp(logits.float().logsumexp(1),ref_logit+math.log(reference_weight))
        ret=(den-logits[torch.arange(len(labels),device=labels.device),labels].float()).mean()
    else:
        ret=F.cross_entropy(logits,labels)
    if arm=='original_written':glob=(1-(out['global_text']*target).sum(-1)).mean();slot=ret*0
    else:
        valid=out['valid'];good=valid.any(1);distance=1-(out['global_text']*target).sum(-1);glob=torch.where(good,distance,torch.zeros_like(distance)).sum()/good.sum().clamp_min(1);distance=1-(out['slots']*teacher).sum(-1);slot=torch.where(valid,distance,torch.zeros_like(distance)).sum()/valid.sum().clamp_min(1)
    return dict(total=ret+.1*glob+.1*slot,retrieval=ret,global_alignment=glob,slot_alignment=slot)
