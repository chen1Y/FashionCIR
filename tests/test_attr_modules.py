from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from attr_slot_data import slot_view,collate_slot_views,SLOTS
try:
    import torch
    from torch import nn
    from attr_slot_fusion import MultiAttrTextEncoder,VisualConflictEraser,SlotDynamicFusion,CIRDisentangleLosses
except ImportError:
    torch=None


class SlotDataTests(unittest.TestCase):
    def test_native_and_missing(self):
        x=slot_view({'target_attrs':{'color':'navy'},'drop_attrs':''},'darker');self.assertEqual(x['target_attrs']['color'],'navy')
        y=slot_view(None,'original');self.assertTrue(y['fallback']);c=collate_slot_views([x,y]);self.assertEqual(c['attr_valid'][1],[False]*4);self.assertEqual(c['mod_text'][1],'original')
    def test_conflict_trace_without_inventing_drop(self):
        s=dict(status='ok',validation={'valid':True},semantic_validation={'valid':True},structured_edit=dict(add=[dict(attribute='pink',category='color',region='body')],retain=[dict(attribute='black',category='color',region='body')],remove=[]))
        x=slot_view(s,'pink');self.assertEqual(x['target_attrs']['color'],'pink');self.assertEqual(x['drop_attrs'],'');self.assertIn('omit_superseded_retain:black',x['trace'])
    def test_invalid_falls_back(self):self.assertTrue(slot_view({'status':'failed'},'edit')['fallback'])


if torch is not None:
    class Tokenizer:
        def __init__(self):self.encoder={str(i):i for i in range(64)};self.decoder={i:str(i) for i in range(64)};self.cache={};self.sot_token_id=1;self.eot_token_id=2
        def encode(self,text):return [3+sum(w.encode())%60 for w in text.split()]
    class Block(nn.Module):
        def __init__(self,d):super().__init__();self.attn=nn.MultiheadAttention(d,8)
        def forward(self,x,mask):return x+self.attn(x,x,x,attn_mask=mask,need_weights=False)[0]
    class Transformer(nn.Module):
        def __init__(self,d):super().__init__();self.resblocks=nn.ModuleList([Block(d)])
        def forward(self,x,attn_mask=None):
            for b in self.resblocks:x=b(x,attn_mask)
            return x
    class Clip(nn.Module):
        def __init__(self,d=32):
            super().__init__();self.token_embedding=nn.Embedding(64,d);self.positional_embedding=nn.Parameter(torch.randn(77,d)*.01);self.transformer=Transformer(d);self.ln_final=nn.LayerNorm(d);self.text_projection=nn.Parameter(torch.eye(d))


@unittest.skipIf(torch is None,'Torch not installed locally; run server suite')
class AttrModuleTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42);torch.set_num_threads(2)
    def encoder(self,**kw):return MultiAttrTextEncoder(Clip(),Tokenizer(),**kw)
    def attrs(self):return [{s:'navy blue lace long sleeve' for s in SLOTS} for _ in range(4)]
    def test_pack_reserved_tokens_long_and_missing(self):
        e=self.encoder();a=self.attrs();a[0]['color']='long '*200;a[1]['shape']='';p=e.pack(a);self.assertEqual(tuple(p['tokens'].shape),(4,77));self.assertTrue(p['truncated'][0,0]);self.assertFalse(p['attr_valid'][1,1]);self.assertTrue((p['tokens'][torch.arange(4),p['eos_indices']]==2).all());self.assertTrue((p['align_indices']<p['eos_indices'][:,None]).all())
    def test_block_mask_and_finite_softmax(self):
        e=self.encoder();p=e.pack(self.attrs());m=e.attention_mask(p);self.assertTrue(torch.isfinite(torch.softmax(m,-1)).all())
        for b in range(4):
            for k in range(4):
                i=p['align_indices'][b,k].item();a,z=p['segment_spans'][b,k].tolist();self.assertEqual(set(torch.isfinite(m[b,i]).nonzero().flatten().tolist()),{0,i,*range(a,z+1)})
            end=p['eos_indices'][b].item();self.assertTrue(torch.isfinite(m[b,end,:end+1]).all())
    def test_gradient_only_new_rows_and_explicit_eos(self):
        e=self.encoder();before=e.clip.token_embedding.weight.detach().clone();o=e(self.attrs());self.assertEqual(tuple(o['f_attr_text'].shape),(4,4,32));((o['f_attr_text']*torch.randn_like(o['f_attr_text'])).sum()+(o['f_global_text']*torch.randn_like(o['f_global_text'])).sum()).backward();self.assertTrue((e.embedding.extra.grad.norm(dim=1)>0).all());self.assertTrue(all(p.grad is None for p in e.clip.parameters()));self.assertTrue(torch.equal(before,e.clip.token_embedding.weight));self.assertFalse(torch.equal(o['tokens'].argmax(-1),o['eos_indices']))
    def test_strict_span_and_unblocked_flags(self):
        e=self.encoder(strict_segment_isolation=True);p=e.pack(self.attrs());m=e.attention_mask(p);a,z=p['segment_spans'][0,2].tolist();self.assertFalse(torch.isfinite(m[0,z,1:a]).any())
        u=self.encoder(use_block_mask=False);p=u.pack(self.attrs());m=u.attention_mask(p);i=p['align_indices'][0,2].item();self.assertTrue(torch.isfinite(m[0,i,:i+1]).all())
    def test_visual_shape_doc_dimensions_and_gradients(self):
        e=VisualConflictEraser(512);p=torch.randn(4,256,512);v=torch.randn(4,512);d=torch.randn(4,512);o=e(p,v,d,torch.tensor([True,False,True,False]));self.assertEqual(tuple(o['f_attr_vis'].shape),(4,4,512));self.assertTrue(torch.equal(o['keep_weights'][1],torch.ones(256)));o['f_attr_vis'].sum().backward();self.assertTrue(torch.isfinite(e.slot_queries.grad).all())
    def test_global_positive_erasure_and_empty_drop(self):
        e=VisualConflictEraser(32);v=torch.nn.functional.normalize(torch.randn(3,32),dim=-1);drop=torch.stack([v[0],-v[1],v[2]]);o=e(torch.randn(3,16,32),v,drop,torch.tensor([True,True,False]));self.assertLess(float(o['v_clean'][0].norm()),1e-5);self.assertTrue(torch.allclose(o['v_clean'][1:],v[1:],atol=1e-6));self.assertTrue(torch.equal(o['keep_weights'][2],torch.ones(16)))
    def test_missing_slots_fallback(self):
        f=SlotDynamicFusion(32);v=torch.randn(4,4,32);t=torch.full_like(v,float('nan'));g=torch.randn(4,32);m=torch.randn(4,32);o=f(v,t,g,m,torch.randn(4,32),torch.zeros(4,4,dtype=torch.bool));self.assertTrue(torch.allclose(o['q_final'],torch.nn.functional.normalize(g+m,dim=-1)));self.assertEqual(float(o['alpha'].sum()),0.)
    def test_losses_masked_nan_padding_and_empty_drop(self):
        loss=CIRDisentangleLosses();q=torch.randn(4,32,requires_grad=True);t=torch.randn(4,3,32);t[:,2]=float('nan');mask=torch.tensor([[True,True,False]]*4);a=torch.randn(4,4,32,requires_grad=True);o=loss(q,t,a,torch.zeros(4,32),torch.zeros(4,dtype=torch.bool),torch.ones(4,4,dtype=torch.bool),mask);o['total'].backward();self.assertTrue(torch.isfinite(q.grad).all());self.assertEqual(float(o['erase']),0.)
    def test_orthogonality_analytic(self):
        l=CIRDisentangleLosses();q=torch.randn(4,8);t=torch.randn(4,8);a=torch.eye(8)[:4][None].repeat(4,1,1);v=torch.ones(4,4,dtype=torch.bool);o=l(q,t,a,torch.zeros(4,8),torch.zeros(4,dtype=torch.bool),v);self.assertEqual(float(o['orthogonality']),0.)
        a=torch.ones(4,4,8);o=l(q,t,a,torch.zeros(4,8),torch.zeros(4,dtype=torch.bool),v);self.assertAlmostEqual(float(o['orthogonality']),1.,places=6)


if __name__=='__main__':unittest.main()
