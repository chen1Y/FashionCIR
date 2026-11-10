import unittest,sys,tempfile,json,hashlib
from unittest import mock
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fiq_full_partition_024 import partition,validation_gate_state
try:
    import torch
except ImportError:
    torch=None

class FullPartitionTests(unittest.TestCase):
    def test_failed_validation_blocks_even_with_stale_allow_flag(self):
        self.assertEqual(validation_gate_state(dict(status='failed',downstream_allowed=True)),'blocked')
        self.assertEqual(validation_gate_state(dict(status='cancelled',downstream_allowed=True)),'blocked')
        self.assertEqual(validation_gate_state(dict(status='online_official_texts',downstream_allowed=True)),'waiting')
        self.assertEqual(validation_gate_state(dict(status='complete',downstream_allowed=False)),'waiting')
        self.assertEqual(validation_gate_state(dict(status='complete',downstream_allowed=True)),'audit_required')
    def test_full_paid_batch_has_sufficient_conservative_allowance(self):
        from cir_cumulative_budget import micro
        self.assertLessEqual(micro((16384*.2+512*.8)*1000/1e6),micro(5))
    def row(self,a,b,i=0):return dict(candidate=a,target=b,source_index=i,modification_text='longer sleeves')
    def test_queries_and_exact_aliases_stay_in_one_partition(self):
        ids=['a','b','alias','c','d'];h=dict(a='1',alias='1',b='2',c='3',d='4');p=partition(ids,h,[self.row('a','b'),self.row('alias','c',1)],[])
        side=next(s for s in p['banks'] if 'a' in p['banks'][s]);self.assertTrue({'a','alias','b','c'}<=set(p['banks'][side]));self.assertEqual(sum(len(x) for x in p['selected'].values()),2);self.assertEqual(p['intersections'],dict(fit_dev=0,fit_heldout=0,dev_heldout=0))
    def test_heldout_and_same_photo_masks_retain_audit_rows(self):
        p=partition(['a','b','alias'],dict(a='1',b='2',alias='1'),[self.row('a','b'),self.row('a','alias',1)],['2'])
        self.assertEqual(len(p['audit']),2);self.assertFalse(any(r['eligible'] for r in p['audit']));self.assertEqual(p['reason_counts']['target_official_heldout_byte'],1);self.assertEqual(p['reason_counts']['reference_target_same_jpeg_ambiguity'],1)
    def test_actual_gt_alias_remains_in_bank_but_negatives_canonical(self):
        p=partition(['alias','gt','ref'],dict(alias='1',gt='1',ref='2'),[self.row('ref','gt')],[]);s=next(s for s in p['banks'] if 'gt' in p['banks'][s]);self.assertIn('gt',p['banks'][s]);self.assertNotIn('gt',p['canonical'][s]);self.assertEqual(p['selected'][s][0]['target'],'gt')
    def test_all_rows_used_without_fit_or_dev_cap(self):
        rows=[self.row('a','b',i) for i in range(1801)];p=partition(['a','b'],dict(a='1',b='2'),rows,[]);self.assertEqual(sum(len(x) for x in p['selected'].values()),1801);self.assertEqual(len(p['audit']),1801)
    def test_deterministic_under_input_order(self):
        ids=['a','b','c','d'];h=dict(a='1',b='2',c='3',d='4');rows=[self.row('a','b'),self.row('c','d',1)];a=partition(ids,h,rows,[]);b=partition(ids,h,list(reversed(rows)),[])
        self.assertEqual(a['banks'],b['banks']);self.assertEqual({r['source_index']:r['split'] for r in a['audit']},{r['source_index']:r['split'] for r in b['audit']})
    def test_missing_and_duplicate_sources_rejected(self):
        with self.assertRaises(ValueError):partition(['a','a'],dict(a='1'),[],[])
        with self.assertRaises(ValueError):partition(['a'],dict(a='1'),[self.row('a','missing')],[])

@unittest.skipIf(torch is None,'Server Torch required')
class OriginalArmInvariance(unittest.TestCase):
    def test_original_loss_does_not_use_attribute_teacher(self):
        from expanded_attr_model import losses
        q=torch.nn.functional.normalize(torch.randn(2,4),dim=-1);g=torch.nn.functional.normalize(torch.randn(3,4),dim=-1);out={'global_text':q};labels=torch.tensor([0,1]);mask=torch.ones(2,3,dtype=torch.bool)
        a=losses(out,q,g,labels,mask,torch.zeros(2,4,4),'original');b=losses(out,q,g,labels,mask,torch.full((2,4,4),float('nan')),'original')
        self.assertTrue(torch.equal(a['total'],b['total']));self.assertEqual(float(a['slot_alignment']),0.)
    def test_complete_reuse_performs_no_clip_loading_or_encoding(self):
        import fiq_full_assets_024 as assets
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as t:
            root=Path(t);(root/'metadata').mkdir();base=root/'base';base.write_bytes(b'base');meta=dict(base_sha256=hashlib.sha256(b'base').hexdigest(),gallery_ids=['a','b'],image_sha256=dict(a='sha-a',b='sha-b'))
            (root/'metadata/shirt_val.json').write_text(json.dumps(meta));features=dict(a=torch.tensor([1.,0.]),b=torch.tensor([0.,1.]))
            with mock.patch.object(assets,'CACHE',root/'cache'),mock.patch.object(assets,'base_path',return_value=base),mock.patch.object(assets,'reuse018',return_value=(features,{},{})),mock.patch.object(assets,'require_space'),mock.patch.object(assets,'load_clip',side_effect=AssertionError('Reencoding is forbidden')):
                assets.prepare_images(root,'shirt','val')
            report=json.loads((root/'assets_shirt_val.json').read_text());self.assertEqual(report['counts']['new_image_forwards'],0);self.assertEqual(report['counts']['reused_global_images'],2)
    def test_weight_router_preserves_values_and_refuses_overwrite(self):
        import train_fiq_full_024 as full
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as t:
            root=Path(t);weights=root/'memory_weights';(root/'plan.json').write_text(json.dumps(dict(weight_directory=str(weights))));path=root/'fits/dress/original/epoch_0_delta.pt';path.parent.mkdir(parents=True)
            with mock.patch.multiple(full.engine,CACHE=full.engine.CACHE,base_path=full.engine.base_path,guard=full.engine.guard,load_context=full.engine.load_context,atomic_torch_save=full.engine.atomic_torch_save),mock.patch.multiple(full.native,base_path=full.native.base_path,load_context=full.native.load_context):
                full.configure(root,'original',False);full.engine.atomic_torch_save(dict(value=torch.arange(4)),path);self.assertTrue(path.is_symlink());self.assertTrue(torch.equal(torch.load(path,weights_only=True)['value'],torch.arange(4)))
                with self.assertRaises(FileExistsError):full.engine.atomic_torch_save(dict(value=torch.zeros(4)),path)

if __name__=='__main__':unittest.main()
