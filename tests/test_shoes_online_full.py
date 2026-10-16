import concurrent.futures
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import generate_shoes_online as online
from generate_shoes_online import Ledger, BudgetStop, UncertainRequest, quality_errors, checked_complete
from run_cirr_shoes_benchmark import run_with_secret


class FullOnlineTests(unittest.TestCase):
    def setUp(self):
        base = Path(tempfile.gettempdir()) if os.name=='posix' else Path(__file__).resolve().parents[1]/'.autodl/test_tmp'
        base.mkdir(parents=True,exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="fashioncir-online-test-",dir=base)
        self.root = Path(self.temp.name).resolve()
        assert self.root.parent == base.resolve() and self.root.name.startswith('fashioncir-online-test-')
        self.addCleanup(self.temp.cleanup)
        self.config = {"model":"test-model", "version":"frozen"}

    def ledger(self, **kwargs):
        return Ledger(self.root/'requests.sqlite3', self.config, **kwargs)

    def test_actual_usage_releases_unused_allowance_without_resetting_spend(self):
        ledger = self.ledger(max_cny=.01)
        for i in range(8):
            ledger.begin('train', i, 'generate'); ledger.reserve(768)
            ledger.account(dict(prompt_tokens=1000, completion_tokens=500)); ledger.finish({'text':'done'})
        summary = ledger.summary()
        self.assertEqual(summary['calls'],8)
        self.assertAlmostEqual(summary['accounted_upper_cny'],.0048)
        ledger.db.close()

    def test_concurrent_reservations_never_exceed_request_cap(self):
        ledger=self.ledger(max_calls=5)
        def call(i):
            try:
                ledger.begin('train',i,'generate');ledger.reserve(768)
                ledger.account(dict(prompt_tokens=1,completion_tokens=1));ledger.finish({'text':'done'})
                return True
            except BudgetStop:
                return False
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(sum(pool.map(call,range(30))),5)
        self.assertEqual(ledger.summary()['calls'],5)
        ledger.db.close()

    def test_unknown_request_is_charged_and_not_blindly_repeated_after_restart(self):
        ledger=self.ledger();ledger.begin('val',58,'generate');ledger.reserve(768)
        before=ledger.summary()['accounted_upper_cny'];ledger.db.close()
        resumed=self.ledger()
        with self.assertRaises(UncertainRequest):resumed.begin('val',58,'generate')
        self.assertEqual(resumed.summary()['accounted_upper_cny'],before)
        self.assertEqual(resumed.summary()['request_states'],{'unknown':1})
        resumed.db.close()

    def test_response_and_metering_survive_restart_and_cache_is_split_bound(self):
        ledger=self.ledger();ledger.begin('train',58,'generate');ledger.reserve(768)
        ledger.account(dict(prompt_tokens=100,completion_tokens=20));ledger.finish({'text':'cached'})
        ledger.db.close();ledger=self.ledger()
        self.assertEqual(ledger.begin('train',58,'generate'),{'text':'cached'})
        self.assertIsNone(ledger.begin('val',58,'generate'))
        self.assertEqual(ledger.summary()['calls'],1)
        ledger.db.close()

    def test_unknown_usage_retains_allowance_and_blocks_cost_overrun(self):
        ledger=self.ledger(max_cny=.005)
        ledger.begin('train',1,'generate');ledger.reserve(768);ledger.account(None);ledger.finish({'text':'done'})
        ledger.begin('train',2,'generate')
        with self.assertRaises(BudgetStop):ledger.reserve(768)
        self.assertEqual(ledger.summary()['calls'],1)
        self.assertEqual(ledger.summary()['usage_missing_calls'],1)
        ledger.db.close()

    def test_resume_cannot_change_protocol_or_raise_budget_silently(self):
        ledger=self.ledger();ledger.db.close()
        with self.assertRaises(ValueError):Ledger(self.root/'requests.sqlite3',{'version':'changed'})
        with self.assertRaises(ValueError):self.ledger(max_cny=16)

    def test_color_gate_rejects_conflicting_strap_but_preserves_unaffected_sole(self):
        item=lambda a,r:dict(attribute=a,category='color',region=r,evidence='reference_image')
        self.assertTrue(quality_errors('are solid red',{'retain':[item('pink strap','upper strap')]}))
        self.assertFalse(quality_errors('are solid red',{'retain':[item('white sole','outsole')]}))
        self.assertFalse(quality_errors('have a white biker design on the side',{'retain':[item('black color','upper')]}))
        self.assertTrue(quality_errors('is black with silver accents',{'retain':[item('light blue','upper')]}))

    def test_completed_failed_queries_are_retained_and_full_coverage_is_required(self):
        rows=[dict(source_index=i,candidate=f'r{i}',target=f't{i}',modification_text='blue') for i in (58,71)]
        samples=[dict(r,query_index=r['source_index'],status='ok' if j==0 else 'failed') for j,r in enumerate(rows)]
        payload=dict(samples=samples,attempts_complete=True)
        self.assertEqual(len(checked_complete(payload,rows)),2)
        with self.assertRaises(ValueError):checked_complete(dict(samples=samples[:1],attempts_complete=False),rows)
        with self.assertRaises(ValueError):checked_complete(payload,rows,min_coverage=.75)
        samples[0]['target']='wrong'
        with self.assertRaises(ValueError):checked_complete(payload,rows)

    @unittest.skipUnless(os.name=='posix','Anonymous subprocess FD inheritance uses Linux')
    def test_secret_pipe_passes_no_credential_in_environment_or_caller_arguments(self):
        command=[sys.executable,'-c',"import os,sys; key=os.fdopen(int(sys.argv[-1])).read(); assert len(key)==18; assert key not in str(os.environ); print('secret_pipe_ok')"]
        with tempfile.TemporaryFile(mode='w+') as log:
            self.assertEqual(run_with_secret(command,'dummy-memory-value',log),0)
            log.seek(0);self.assertEqual(log.read().strip(),'secret_pipe_ok')
        self.assertNotIn('dummy-memory-value',' '.join(command))

    @unittest.skipUnless(os.name=='posix','Generator process lock uses Linux')
    def test_fake_api_full_export_resume_and_target_exclusion(self):
        rows={s:[dict(source_index=i,candidate=f'ref{i}',target=f'TARGET-HIDDEN-{i}',modification_text='is blue') for i in ids]
              for s,ids in [('train',(58,71)),('val',(236,))]}
        meta=SimpleNamespace(queries=rows,image_paths={r['candidate']:self.root/'unused.jpg' for rs in rows.values() for r in rs})
        edit=dict(retain=[],remove=[],add=[dict(attribute='blue',category='color',region='upper',evidence='modification_text')],
                  relations=[],target_description='A blue shoe',complexity=dict(level='simple',score=.1,reason='recolor'),confidence=dict(score=.8,ambiguities=[]))
        class FakeClient:
            texts=[]
            def __init__(self,key,endpoint,budget):self.key,self.budget=key,budget
            def safe_text(self,text):return str(text).replace(self.key,'[REDACTED]')
            def ask(self,image,text,tokens):
                self.texts.append(text);self.budget.reserve(tokens)
                self.budget.account(dict(prompt_tokens=100,completion_tokens=20))
                return dict(text=json.dumps(edit if tokens==768 else dict(valid=True,reasons=[])),usage=dict(prompt_tokens=100,completion_tokens=20),model='fake')
        def run(split):
            read_fd,write_fd=os.pipe();os.write(write_fd,b'dummy-memory-value');os.close(write_fd)
            argv=['program','--root',str(self.root),'--split',split,'--output',str(self.root/f'{split}.json'),'--ledger',str(self.root/'requests.sqlite3'),'--workers','2','--api-key-fd',str(read_fd)]
            with patch.object(sys,'argv',argv),patch.object(online,'BenchmarkData',return_value=meta),patch.object(online,'Client',FakeClient),patch.object(online,'reference_data_url',return_value='data:image/jpeg;base64,REFERENCE'),patch.object(online.RateLimiter,'wait'),patch.object(online,'require_space',return_value=20):
                online.main()
        run('train');run('val');before=len(FakeClient.texts);run('train')
        self.assertEqual(before,6);self.assertEqual(len(FakeClient.texts),6)
        self.assertNotIn('TARGET-HIDDEN','\n'.join(FakeClient.texts))
        for split in ('train','val'):
            payload=json.loads((self.root/f'{split}.json').read_text())
            self.assertTrue(payload['attempts_complete']);self.assertEqual(payload['valid_queries'],len(rows[split]))


if __name__=='__main__':unittest.main()
