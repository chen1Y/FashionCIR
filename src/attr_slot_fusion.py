"""Standalone attribute aggregation and conflict-erased fusion for OpenCLIP.

    Existing checkpoints/tokenizer objects are never mutated. Attention isolation
    follows the supplied specification; optional strict spans also block indirect
    cross-span paths. Patch tokens must be projected into the CLIP joint space.
"""
from __future__ import annotations
import copy
import math
from typing import Any
import torch
from torch import Tensor,nn
from torch.nn import functional as F
from attr_slot_data import SLOTS

ALIGN_TOKENS=('<|align_color|>','<|align_shape|>','<|align_detail|>','<|align_tex|>')


class ExpandedFrozenEmbedding(nn.Module):
    """Virtual vocabulary expansion with only four trainable rows.

    Keeping base rows as frozen parameters avoids AdamW weight-decay drift and
    allocating optimizer states for the entire original vocabulary.
    """
    def __init__(self,base:nn.Embedding)->None:
        super().__init__();self.base=base;self.base.requires_grad_(False)
        self.extra=nn.Parameter(base.weight.detach().mean(0,keepdim=True).repeat(4,1).clone())
        self.num_embeddings=base.num_embeddings+4;self.embedding_dim=base.embedding_dim

    def forward(self,tokens:Tensor)->Tensor:
        if ((tokens<0)|(tokens>=self.num_embeddings)).any():raise ValueError('Token outside expanded vocabulary')
        n=self.base.num_embeddings;ordinary=self.base(tokens.clamp_max(n-1));extra=F.embedding((tokens-n).clamp(0,3),self.extra)
        return torch.where((tokens>=n)[...,None],extra,ordinary)


class MultiAttrTextEncoder(nn.Module):
    """Four aggregation tokens plus explicit EOS pooling, compatible with OpenCLIP2.23.

    Each special row sees SOS, its own inclusive span, and itself. Standard rows
    remain causal unless strict_segment_isolation is requested. EOS sees all real
    preceding tokens. Special IDs exceed EOS, so argmax-based EOS pooling is never
    used. All token/segment truncations and positions are returned for auditing.
    """
    def __init__(self,clip:nn.Module,tokenizer:Any,*,use_block_mask:bool=True,strict_segment_isolation:bool=False)->None:
        super().__init__();self.clip=clip;clip.requires_grad_(False);self.embedding=ExpandedFrozenEmbedding(clip.token_embedding)
        self.tokenizer=copy.deepcopy(tokenizer);self.context_length=int(clip.positional_embedding.shape[0]);self.use_block_mask=use_block_mask;self.strict_segment_isolation=strict_segment_isolation
        self.special_ids={t:clip.token_embedding.num_embeddings+i for i,t in enumerate(ALIGN_TOKENS)}
        for token,index in self.special_ids.items():
            self.tokenizer.encoder[token]=index
            if hasattr(self.tokenizer,'decoder'):self.tokenizer.decoder[index]=token
            if hasattr(self.tokenizer,'cache'):self.tokenizer.cache[token]=token
        if hasattr(self.tokenizer,'pat'):
            import regex
            self.tokenizer.pat=regex.compile('|'.join(regex.escape(t) for t in ALIGN_TOKENS)+'|'+self.tokenizer.pat.pattern,regex.IGNORECASE)

    def pack(self,attrs:list[dict[str,str]])->dict[str,Tensor]:
        """Reserve all four special tokens and EOS, even for long or missing fields."""
        device=self.embedding.extra.device;length=self.context_length;labels=[self.tokenizer.encode(s.title()+':') for s in SLOTS]
        budget=length-2-4-sum(map(len,labels))
        if budget<4:raise ValueError('Context too short for four labeled slots')
        tokens=[];align=[];spans=[];eos=[];valid=[];truncated=[]
        for row in attrs:
            seq=[self.tokenizer.sot_token_id];a=[];s=[];v=[];tr=[]
            for k,slot in enumerate(SLOTS):
                content=self.tokenizer.encode(str(row.get(slot,'') or '').strip());limit=budget//4+int(k<budget%4)
                start=len(seq);seq+=labels[k]+content[:limit];s.append([start,len(seq)-1]);a.append(len(seq));seq.append(self.special_ids[ALIGN_TOKENS[k]])
                v.append(bool(content));tr.append(len(content)>limit)
            eos.append(len(seq));seq.append(self.tokenizer.eot_token_id);seq += [0]*(length-len(seq));tokens.append(seq);align.append(a);spans.append(s);valid.append(v);truncated.append(tr)
        return dict(tokens=torch.tensor(tokens,device=device,dtype=torch.long),align_indices=torch.tensor(align,device=device),segment_spans=torch.tensor(spans,device=device),
            eos_indices=torch.tensor(eos,device=device),attr_valid=torch.tensor(valid,device=device,dtype=torch.bool),truncated=torch.tensor(truncated,device=device,dtype=torch.bool))

    def attention_mask(self,packed:dict[str,Tensor])->Tensor:
        tokens=packed['tokens'];b,l=tokens.shape;mask=torch.full((b,l,l),-torch.inf,device=tokens.device)
        for i in range(b):
            end=int(packed['eos_indices'][i]);mask[i]=torch.triu(torch.full((l,l),-torch.inf,device=tokens.device),diagonal=1);mask[i,:,end+1:]=-torch.inf
            if self.use_block_mask:
                for k in range(4):
                    idx=int(packed['align_indices'][i,k]);start,stop=packed['segment_spans'][i,k].tolist();mask[i,idx,:]=-torch.inf;mask[i,idx,0]=0;mask[i,idx,start:stop+1]=0;mask[i,idx,idx]=0
                    if self.strict_segment_isolation:
                        for q in range(start,stop+1):mask[i,q,:]=-torch.inf;mask[i,q,0]=0;mask[i,q,start:q+1]=0
            mask[i,end,:end+1]=0
        if not torch.isfinite(mask).any(-1).all():raise ValueError('All-masked attention row')
        return mask

    def forward(self,attrs:list[dict[str,str]])->dict[str,Tensor]:
        packed=self.pack(attrs);mask=self.attention_mask(packed);clip=self.clip
        cast=clip.transformer.get_cast_dtype() if hasattr(clip.transformer,'get_cast_dtype') else self.embedding.extra.dtype
        x=self.embedding(packed['tokens']).to(cast)+clip.positional_embedding.to(cast)
        heads=clip.transformer.resblocks[0].attn.num_heads
        x=clip.transformer(x.permute(1,0,2),attn_mask=mask.repeat_interleave(heads,dim=0)).permute(1,0,2)
        x=clip.ln_final(x);projection=clip.text_projection
        x=projection(x) if isinstance(projection,nn.Module) else x@projection
        batch=torch.arange(len(attrs),device=x.device)
        attr=x[batch[:,None],packed['align_indices']];global_text=x[batch,packed['eos_indices']]
        return dict(f_attr_text=F.normalize(attr.float(),dim=-1),f_global_text=F.normalize(global_text.float(),dim=-1),**packed)


class VisualConflictEraser(nn.Module):
    """Positive global projection subtraction and conflict-suppressed slot pooling."""
    def __init__(self,dimension:int,slots:int=4,heads:int=8,temperature:float=10.,use_conflict_eraser:bool=True)->None:
        super().__init__();self.gamma=nn.Parameter(torch.tensor(1.));self.temperature=temperature;self.use_conflict_eraser=use_conflict_eraser
        self.slot_queries=nn.Parameter(torch.randn(slots,dimension)/math.sqrt(dimension));self.cross_attention=nn.MultiheadAttention(dimension,heads,batch_first=True)

    def forward(self,patches:Tensor,v_cls:Tensor,t_drop:Tensor,has_drop:Tensor)->dict[str,Tensor]:
        v=F.normalize(v_cls.float(),dim=-1);drop=F.normalize(t_drop.float(),dim=-1);active=has_drop.bool() & self.use_conflict_eraser
        clean=v-self.gamma*F.relu((v*drop).sum(-1,keepdim=True))*drop*active[:,None]
        score=(F.normalize(patches.float(),dim=-1)*drop[:,None]).sum(-1)
        keep=torch.where(active[:,None],1-torch.sigmoid(self.temperature*score),torch.ones_like(score))
        suppressed=patches.float()*keep[...,None];queries=self.slot_queries[None].expand(len(v),-1,-1)
        slots,_=self.cross_attention(queries,suppressed,suppressed,need_weights=False)
        return dict(v_clean=clean,f_attr_vis=F.normalize(slots.float(),dim=-1),keep_weights=keep)


class SlotDynamicFusion(nn.Module):
    """Shared per-slot replacement gate and learnable global balancing weights."""
    def __init__(self,dimension:int)->None:
        super().__init__();self.gate=nn.Sequential(nn.Linear(3*dimension,dimension//2),nn.GELU(),nn.Linear(dimension//2,1),nn.Sigmoid())
        self.beta1=nn.Parameter(torch.tensor(1.));self.beta2=nn.Parameter(torch.tensor(1.))

    def forward(self,visual:Tensor,text:Tensor,v_clean:Tensor,t_mod:Tensor,global_text:Tensor,attr_valid:Tensor)->dict[str,Tensor]:
        text=torch.where(attr_valid[...,None],text,torch.zeros_like(text))
        mod=t_mod[:,None].expand_as(visual);alpha=self.gate(torch.cat([visual,text,mod],-1));alpha=alpha*attr_valid[...,None]
        slots=alpha*text+(1-alpha)*visual;query=self.beta1*v_clean+self.beta2*global_text+slots.mean(1)
        fallback=~attr_valid.any(1);query=torch.where(fallback[:,None],v_clean+t_mod,query)
        return dict(q_final=F.normalize(query.float(),dim=-1),F_composed_slots=slots,alpha=alpha)


class CIRDisentangleLosses(nn.Module):
    """Candidate/batch asymmetric InfoNCE plus masked orthogonality and erasing."""
    def __init__(self,temperature:float=.07,lambda_ortho:float=.01,lambda_erase:float=.01,margin:float=0.,use_ortho_loss:bool=True)->None:
        super().__init__();self.temperature=temperature;self.lambda_ortho=lambda_ortho;self.lambda_erase=lambda_erase;self.margin=margin;self.use_ortho_loss=use_ortho_loss

    def forward(self,q:Tensor,targets:Tensor,attrs:Tensor,drop:Tensor,has_drop:Tensor,attr_valid:Tensor,candidate_mask:Tensor|None=None)->dict[str,Tensor]:
        if candidate_mask is not None and targets.ndim==3:targets=torch.where(candidate_mask[...,None],targets,torch.zeros_like(targets))
        q=F.normalize(q.float(),dim=-1);targets=F.normalize(targets.float(),dim=-1)
        logits=q@targets.T if targets.ndim==2 else torch.einsum('bd,bmd->bm',q,targets)
        labels=torch.arange(len(q),device=q.device) if targets.ndim==2 else torch.zeros(len(q),device=q.device,dtype=torch.long)
        if candidate_mask is not None:
            if not candidate_mask[:,0].all() or not (candidate_mask.sum(1)>=2).all():raise ValueError('Invalid positive/negative mask')
            logits=logits.masked_fill(~candidate_mask,-torch.inf)
        ret=F.cross_entropy(logits/self.temperature,labels);attrs=F.normalize(torch.where(attr_valid[...,None],attrs,torch.zeros_like(attrs)).float(),dim=-1);gram=attrs@attrs.transpose(1,2)
        pairs=attr_valid[:,:,None]&attr_valid[:,None,:]&~torch.eye(attrs.shape[1],device=q.device,dtype=torch.bool)[None]
        ortho=torch.where(pairs,gram.square(),torch.zeros_like(gram)).sum()/pairs.sum().clamp_min(1)
        erase=F.relu((q*F.normalize(drop.float(),dim=-1)).sum(-1)-self.margin);erase=(erase*has_drop).sum()/has_drop.sum().clamp_min(1)
        total=ret+(self.lambda_ortho*ortho if self.use_ortho_loss else 0)+self.lambda_erase*erase
        return dict(total=total,retrieval=ret,orthogonality=ortho,erase=erase)


class AttrSlotCIR(nn.Module):
    """Protected-CLIP integration point, consuming cached projected visual tokens."""
    def __init__(self,clip:nn.Module,tokenizer:Any,*,use_block_mask:bool=True,use_conflict_eraser:bool=True,use_ortho_loss:bool=True)->None:
        super().__init__();dimension=int(clip.text_projection.shape[-1]);self.text=MultiAttrTextEncoder(clip,tokenizer,use_block_mask=use_block_mask)
        self.eraser=VisualConflictEraser(dimension,use_conflict_eraser=use_conflict_eraser);self.fusion=SlotDynamicFusion(dimension)
        self.losses=CIRDisentangleLosses(use_ortho_loss=use_ortho_loss,lambda_erase=.01 if use_conflict_eraser else 0.)

    def train(self,mode:bool=True)->'AttrSlotCIR':
        super().train(mode);self.text.clip.eval();return self

    def forward(self,attrs:list[dict[str,str]],patches:Tensor,v_cls:Tensor,t_mod:Tensor,t_drop:Tensor,has_drop:Tensor)->dict[str,Tensor]:
        text=self.text(attrs);visual=self.eraser(patches,v_cls,t_drop,has_drop)
        global_text=torch.where(text['attr_valid'].any(1)[:,None],text['f_global_text'],t_mod)
        fused=self.fusion(visual['f_attr_vis'],text['f_attr_text'],visual['v_clean'],t_mod,global_text,text['attr_valid'])
        return dict(**fused,**text,**visual)
