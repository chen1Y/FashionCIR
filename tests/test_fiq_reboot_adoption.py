from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from queue_fiq_research_after_reboot import check_completed_ledger


class AdoptionTests(unittest.TestCase):
    def fixture(self):return dict(status='awaiting_next_research_plan',completed_plans=['p'],jobs=[dict(id='j',plan='p',status='complete',returncode=0)]),{'p':dict(id='p',jobs=[dict(id='j')])}
    def test_complete_only(self):s,p=self.fixture();check_completed_ledger(s,p)
    def test_inflight_rejected(self):
        s,p=self.fixture();s['jobs'][0]['status']='running'
        with self.assertRaises(ValueError):check_completed_ledger(s,p)
    def test_failed_rejected(self):
        s,p=self.fixture();s['jobs'][0]['returncode']=1
        with self.assertRaises(ValueError):check_completed_ledger(s,p)
    def test_unpublished_work_rejected(self):
        s,p=self.fixture();p['q']=dict(id='q',jobs=[])
        with self.assertRaises(ValueError):check_completed_ledger(s,p)
    def test_duplicate_history_rejected(self):
        s,p=self.fixture();s['jobs'].append(s['jobs'][0].copy())
        with self.assertRaises(ValueError):check_completed_ledger(s,p)
    def test_missing_job_rejected(self):
        s,p=self.fixture();s['jobs']=[]
        with self.assertRaises(ValueError):check_completed_ledger(s,p)


if __name__=='__main__':unittest.main()
