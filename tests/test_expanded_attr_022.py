import json,sqlite3,tempfile,unittest,hashlib
from pathlib import Path
from expanded_attr_schema_021 import PROMPT,request_key
from expanded_attr_resume_022 import reconcile,publish_ready,transport_fallback_allowed

class ReconcileTests(unittest.TestCase):
    def test_transport_bound_and_authentication_stop(self):
        self.assertTrue(transport_fallback_allowed('Official API transport error (URLError); no automatic retry',1))
        self.assertTrue(transport_fallback_allowed('Official API transport error (TimeoutError); no automatic retry',3))
        self.assertFalse(transport_fallback_allowed('Official API transport error (URLError)',4))
        self.assertFalse(transport_fallback_allowed('Official API HTTP 401',1))
    def make(self,base,state='known',returned=True,prompt=None):
        d=base/'generation'/'train_000';d.mkdir(parents=True);k=request_key('sha','longer sleeves')
        row=dict(request_key=k,request_index=0,reference_sha256='sha',instruction='longer sleeves',category='shirt')
        response=dict(text='saved',usage={},finish_reason='stop') if returned else None
        payload=dict(status='fallback',target_attrs={x:'' for x in ('color','shape','detail','texture')},errors=['invalid'],response=response,request_key=k,request_index=0)
        (d/'protocol.json').write_text(json.dumps(dict(model='qwen3.7-flash',prompt_sha256=prompt or hashlib.sha256(PROMPT.encode()).hexdigest(),rows=[row])))
        (d/'responses.json').write_text(json.dumps([payload]));db=sqlite3.connect(d/'requests.sqlite3');db.execute('CREATE TABLE requests(idx INTEGER,state TEXT,response TEXT)')
        if state:db.execute('INSERT INTO requests VALUES(0,?,?)',(state,json.dumps(response) if response else None))
        db.commit();db.close();return d,k
    def test_known_invalid_reused_without_paid_repair(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as t:
            d,k=self.make(Path(t));a,p=reconcile(Path(t));self.assertIn(k,a);self.assertEqual(a[k]['status'],'fallback');self.assertEqual(p['batches'][0]['known'],1)
    def test_unknown_never_resubmitted(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as t:
            d,k=self.make(Path(t),'unknown',False);a,p=reconcile(Path(t));self.assertEqual(a[k]['errors'],['unknown021_no_resend'])
    def test_unsent_eligible_not_mislabeled_unknown(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as t:
            d,k=self.make(Path(t),None,False);a,p=reconcile(Path(t));self.assertNotIn(k,a);self.assertEqual(p['unsent_keys'],[k])
    def test_inflight_refused(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as t:
            self.make(Path(t),'reserved',False)
            with self.assertRaises(ValueError):reconcile(Path(t))
    def test_wrong_prompt_refused(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as t:
            self.make(Path(t),prompt='wrong')
            with self.assertRaises(ValueError):reconcile(Path(t))
    def test_ledger_conflict_refused(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as t:
            d,k=self.make(Path(t));db=sqlite3.connect(d/'requests.sqlite3');db.execute("UPDATE requests SET response='{}'");db.commit();db.close()
            with self.assertRaises(ValueError):reconcile(Path(t))
    def test_publish_ready_once_and_no_partial_query_drop(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as t:
            root=Path(t);rows=[dict(category='shirt',request_key='a'),dict(category='toptee',request_key='b')];a={'a':dict(status='ok',target_attrs=dict(color='white'),errors=[])}
            publish_ready(root,'train',rows,a);p=root/'texts/shirt_train.json';before=p.stat().st_mtime_ns
            self.assertFalse((root/'texts/toptee_train.json').exists());publish_ready(root,'train',rows,a);self.assertEqual(before,p.stat().st_mtime_ns)
            a['a']['target_attrs']['color']='black'
            with self.assertRaises(ValueError):publish_ready(root,'train',rows,a)

if __name__=='__main__':unittest.main()
