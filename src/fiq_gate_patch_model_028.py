"""028 learned attribute weights; other image/attribute fusion coefficients fixed."""
import torch
from torch import nn
from torch.nn import functional as F
from attr_token_pair_alignment import PairAlignedText
from expanded_attr_model import branch_forward as original_branch,compose as original_compose


class AttributeGate(nn.Module):
    def __init__(self):
        super().__init__();self.mlp=nn.Sequential(nn.Linear(7,16),nn.GELU(),nn.Linear(16,1));nn.init.zeros_(self.mlp[-1].weight);nn.init.zeros_(self.mlp[-1].bias)
    def forward(self,slots,eos,reference,valid):
        s=slots.float();e=eos.float();v=reference.float();identity=torch.eye(4,device=s.device).expand(len(s),-1,-1)
        features=torch.cat([(s*v[:,None]).sum(-1,keepdim=True),(s*e[:,None]).sum(-1,keepdim=True),(v*e).sum(-1)[:,None,None].expand(-1,4,-1),identity],-1)
        logits=self.mlp(features).squeeze(-1).float();any_valid=valid.any(-1,keepdim=True);safe=torch.where(any_valid,logits.masked_fill(~valid,-torch.inf),torch.zeros_like(logits));return safe.softmax(-1)*valid


class GatedAttributeText(PairAlignedText):
    def __init__(self,clip,tokenizer,last_blocks=2):
        super().__init__(clip,tokenizer,last_blocks);self.attribute_gate=AttributeGate()
    def parameter_groups(self):return super().parameter_groups()+[dict(params=list(self.attribute_gate.parameters()),lr=2e-4,weight_decay=.01)]
    def delta_state(self):
        state=super().delta_state();state['attribute_gate']={k:v.detach().cpu().clone() for k,v in self.attribute_gate.state_dict().items()};return state
    def load_delta(self,state):
        super().load_delta(state)
        if 'attribute_gate' in state:self.attribute_gate.load_state_dict(state['attribute_gate'])


def branch_forward(model,records,arm):
    if arm!='slots':raise ValueError('028 gate is preregistered for image+slots only')
    out=original_branch(model,records,arm);out['attribute_gate']=model.attribute_gate;return out


def compose(out,reference,arm):
    if arm!='slots':raise ValueError('028 gated arm must be slots')
    valid=out['valid'];weights=out['attribute_gate'](out['slots'],out['global_text'],reference,valid);average=(out['slots']*weights[...,None]).sum(1);generated=F.normalize(out['global_text']+average,dim=-1);out['gate_weights']=weights;out['text']=torch.where(valid.any(1)[:,None],generated,out['text']);return F.normalize(reference.float()+out['text'],dim=-1)


def local_scores(slots,valid,patches):
    """Return exact valid-slot sum_i max_j cosine, plus winners for auditing."""
    s=F.normalize(slots.float(),dim=-1);p=F.normalize(patches.float(),dim=-1);b,k,d=s.shape;c,j,width=p.shape
    if width!=d:raise ValueError('Patch/text joint-space dimension mismatch')
    similarity=(s.reshape(b*k,d)@p.reshape(c*j,d).T).reshape(b,k,c,j);maxima,winners=similarity.max(-1);score=torch.where(valid[...,None],maxima,torch.zeros_like(maxima)).sum(1);return score,maxima,winners
