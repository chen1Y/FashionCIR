"""018 text-only paired-image prealignment; never unfreezes the vision encoder."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from attr_slot_fusion import MultiAttrTextEncoder, ALIGN_TOKENS


class HookedVocabulary(nn.Embedding):
    """The requested full vocabulary parameter with a four-row gradient hook.

    Must use a NEW optimizer with zero embedding weight decay: a gradient hook
    alone does not prevent AdamW decay or historical momentum changing old rows.
    """
    def __init__(self, base: nn.Embedding):
        super().__init__(base.num_embeddings+4, base.embedding_dim, device=base.weight.device, dtype=base.weight.dtype)
        self.base_size=base.num_embeddings
        with torch.no_grad():
            self.weight[:self.base_size].copy_(base.weight)
            self.weight[self.base_size:].copy_(base.weight.mean(0,keepdim=True).expand(4,-1))
        self.weight.register_hook(self.only_new_rows)

    @property
    def extra(self):
        # Compatibility view used by the sealed017 packer for dtype/device.
        return self.weight[self.base_size:]

    def only_new_rows(self, gradient):
        out=torch.zeros_like(gradient)
        out[self.base_size:]=gradient[self.base_size:]
        return out


class PairAlignedText(MultiAttrTextEncoder):
    def __init__(self, clip, tokenizer, last_blocks=2):
        super().__init__(clip,tokenizer,use_block_mask=True)
        if last_blocks not in (0,2):raise ValueError('Preregistered arms are token-only and last TWO blocks')
        self.embedding=HookedVocabulary(clip.token_embedding)
        self.last_blocks=last_blocks
        if last_blocks:
            for block in clip.transformer.resblocks[-last_blocks:]:block.requires_grad_(True)
        # Everything else, including visual, original vocabulary, final norm,
        # positional embeddings and text projection remains frozen.
        clip.eval()

    def parameter_groups(self):
        groups=[dict(params=[self.embedding.weight],lr=2e-4,weight_decay=0.)]
        blocks=[p for p in self.clip.parameters() if p.requires_grad]
        if blocks:groups.append(dict(params=blocks,lr=1e-5,weight_decay=.01))
        return groups

    def delta_state(self):
        return dict(new_token_rows=self.embedding.weight[self.embedding.base_size:].detach().cpu().clone(),
                    blocks={n:p.detach().cpu().clone() for n,p in self.clip.named_parameters() if p.requires_grad})

    def load_delta(self,state):
        with torch.no_grad():
            self.embedding.weight[self.embedding.base_size:].copy_(state['new_token_rows'].to(self.embedding.weight))
            for n,p in self.clip.named_parameters():
                if n in state['blocks']:p.copy_(state['blocks'][n].to(p))


def pair_query(out,reference,original_mod):
    valid=out['attr_valid']
    slots=torch.where(valid[...,None],out['f_attr_text'],torch.zeros_like(out['f_attr_text']))
    average=slots.sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
    composed=reference.float()+out['f_global_text']+average
    composed=torch.where(valid.any(1)[:,None],composed,reference.float()+original_mod.float())
    return F.normalize(composed.float(),dim=-1)


def paired_losses(out,reference,original_mod,candidates,candidate_mask,slot_teacher):
    """GT is an actual paired target; slot teacher is a frozen soft visual cue,
    NOT human attribute/region ground truth. Never interpreted as disentanglement.
    """
    q=pair_query(out,reference,original_mod)
    candidates=torch.where(candidate_mask[...,None],candidates,torch.zeros_like(candidates))
    score=torch.einsum('bd,bmd->bm',q,candidates.float()).masked_fill(~candidate_mask,-torch.inf)
    ret=F.cross_entropy(score/.07,torch.zeros(len(q),dtype=torch.long,device=q.device))
    valid=out['attr_valid'];target=candidates[:,0].float()
    global_loss=(1-(out['f_global_text']*target).sum(-1)).mean()
    teacher=F.normalize(torch.where(valid[...,None],slot_teacher,torch.zeros_like(slot_teacher)).float(),dim=-1)
    slot=(1-(out['f_attr_text']*teacher).sum(-1))
    slot=torch.where(valid,slot,torch.zeros_like(slot)).sum()/valid.sum().clamp_min(1)
    return dict(total=ret+.1*global_loss+.1*slot,retrieval=ret,global_alignment=global_loss,slot_alignment=slot)


def packed_audit(model,attrs):
    p=model.pack(attrs);m=model.attention_mask(p);result=[]
    for i,row in enumerate(attrs):
        end=int(p['eos_indices'][i]);ids=p['tokens'][i,:end+1].tolist()
        pieces=[model.tokenizer.decoder.get(t,str(t)) for t in ids]
        readable=[]
        for k,slot in enumerate(('color','shape','detail','texture')):
            a,z=p['segment_spans'][i,k].tolist()
            content=p['tokens'][i,a:z+1].tolist()
            readable.append(model.tokenizer.decode(content)+' '+ALIGN_TOKENS[k])
        result.append(dict(attrs=row,actual_packed_text='[SOS] '+' '.join(readable)+' [EOS]',token_ids=ids,
            bpe_pieces=pieces,align_positions=p['align_indices'][i].tolist(),eos_position=end,
            valid=p['attr_valid'][i].tolist(),truncated=p['truncated'][i].tolist(),
            visible_indices=[torch.isfinite(m[i,int(j)]).nonzero().flatten().tolist() for j in p['align_indices'][i]]))
    return result
