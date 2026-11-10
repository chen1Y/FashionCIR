from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from queue_fiq_after_inspected_failure import inspect_idle


class FailureAuditTests(unittest.TestCase):
    def fixture(self):
        disk=patch('queue_fiq_after_inspected_failure.require_space',return_value=8.5);disk.start();self.addCleanup(disk.stop)
        temporary_root=Path(__file__).resolve().parents[1]/'.autodl';temporary_root.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=temporary_root);r=Path(self.temp.name);assert r.resolve().is_relative_to(temporary_root.resolve());self.addCleanup(self.temp.cleanup);out=r/'results';primary=r/'primary';out.mkdir();primary.mkdir();(out/'plans').mkdir();f=out/'attr_slot_pilot_017/shirt';f.mkdir(parents=True)
        for n in ('protocol.json','manifest.json'):(f/n).write_text('{}')
        state=dict(status='stopped',completed_plans=['old'],jobs=[dict(id='prior',plan='old',status='complete',returncode=0),dict(id='attr_slot_document_shirt',plan='017_attr_slot_document_pilot',status='failed',returncode=1)],error_type='RuntimeError',error='Diagnostic failed; no automatic retry')
        (out/'research_state.json').write_text(json.dumps(state));(out/'queue.pid').write_text('999999999');(out/'attr_slot_document_shirt.log').write_text('TypeError: argsort() received an invalid combination')
        (primary/'experiment_report.json').write_text('{"status":"complete"}');(out/'plans/old.json').write_text(json.dumps(dict(id='old',jobs=[dict(id='prior')])));(out/'plans/017_attr_slot_document_pilot.json').write_text(json.dumps(dict(id='017_attr_slot_document_pilot',jobs=[],dependency_hashes={})))
        return out,primary
    def test_known_prefit_failure_only(self):o,p=self.fixture();s,_=inspect_idle(o,p);self.assertEqual(s['jobs'][-1]['status'],'failed')
    def test_saved_weight_blocks_adoption(self):
        o,p=self.fixture();(o/'attr_slot_pilot_017/shirt/full_head.pt').write_bytes(b'preserve')
        with self.assertRaises(ValueError):inspect_idle(o,p)
    def test_unexpected_error_blocks(self):
        o,p=self.fixture();(o/'attr_slot_document_shirt.log').write_text('different error')
        with self.assertRaises(ValueError):inspect_idle(o,p)
    def test_cancel_blocks(self):
        o,p=self.fixture();(o/'cancel.request').write_text('user cancelled')
        with self.assertRaises(ValueError):inspect_idle(o,p)


if __name__=='__main__':unittest.main()
