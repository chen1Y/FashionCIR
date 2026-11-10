from pathlib import Path
import sys
import tempfile
import unittest
import json
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from queue_fiq_research import seal_plan, parse_reload, ready, compatibility_resume_state


class QueueTests(unittest.TestCase):
    def test_partial_stage_publication_does_not_inspect_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'experiment_report.json').write_text(json.dumps(dict(status='toptee_dqu_original',
                stages=[dict(category='toptee',stage='dqu_original',status='complete',returncode=0)])))
            with patch('queue_fiq_research.build',side_effect=FileNotFoundError('publication in progress')) as build:
                self.assertFalse(ready(root))
                build.assert_not_called()

    def test_final_completion_does_not_hide_missing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'experiment_report.json').write_text(json.dumps(dict(status='complete')))
            with patch('queue_fiq_research.build',side_effect=FileNotFoundError('actually missing')):
                with self.assertRaises(FileNotFoundError):
                    ready(root)

    def test_recovery_rejects_any_started_job(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state = dict(status='stopped',jobs=[dict(id='already_started')],completed_plans=[],
                error_type='FileNotFoundError',error='Completed stage missing metric file: x',pid=1)
            (root/'research_state.json').write_text(json.dumps(state))
            with self.assertRaises(ValueError):
                compatibility_resume_state(root)

    def test_unmetered_paid_job_rejected(self):
        with self.assertRaises(ValueError):
            seal_plan(dict(api_cap_cny=1,jobs=[]))

    def test_oversized_job_rejected(self):
        with self.assertRaises(ValueError):
            seal_plan(dict(api_cap_cny=0,jobs=[dict(id='oversized',max_seconds=3601,command=[sys.executable,'scripts/report_fiq_comparison.py'])]))

    def test_reload_requires_metrics_not_only_exit_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder)/'reload.log'
            log.write_text('done without values')
            with self.assertRaises(ValueError):
                parse_reload(log,'shirt')
            log.write_text('progress\n{"shirt_r1": 20,"shirt_r10":50,"shirt_r50":70}\n')
            self.assertEqual(parse_reload(log,'shirt')['shirt_r10'],50)


if __name__=='__main__':
    unittest.main()
