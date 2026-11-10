import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from report_fiq_comparison import PAPER, MODES, build, markdown


class ComparisonTests(unittest.TestCase):
    def fixture(self, root, complete):
        report = dict(status='complete' if complete else 'shirt_structured_off',stages=[])
        for cat in ('shirt','toptee'):
            (root/cat/'checkpoints').mkdir(parents=True)
            for mode in MODES:
                if complete or (cat,mode)==('shirt','dqu_original'):
                    report['stages'].append(dict(category=cat,stage=mode,status='complete',returncode=0))
                    (root/cat/'checkpoints'/f'{cat}_{mode}_seed42_best.json').write_text(json.dumps({cat+'_r1':25,cat+'_r10':55,cat+'_r50':75}))
            if complete:
                for split in ('train','val'):
                    (root/cat/f'structured_{cat}_{split}.json').write_text(json.dumps(dict(state='complete',completed_queries=100,valid_queries=95)))
        (root/'experiment_report.json').write_text(json.dumps(report))

    def test_pending_is_not_zero_or_final(self):
        with tempfile.TemporaryDirectory() as folder:
            self.fixture(Path(folder),False)
            value = build(folder)
            self.assertFalse(value['final_comparison_complete'])
            row = value['categories']['toptee']['rows'][1]
            self.assertEqual(row['status'],'pending')
            self.assertIsNone(row['values'])

    def test_paper_missing_r1_and_final_pp(self):
        with tempfile.TemporaryDirectory() as folder:
            self.fixture(Path(folder),True)
            value = build(folder)
            self.assertTrue(value['final_comparison_complete'])
            self.assertIsNone(PAPER['shirt']['r1'])
            self.assertEqual(value['categories']['shirt']['rows'][2]['delta_vs_server_pp']['r10'],0)
            self.assertIn('—',markdown(value))

    def test_complete_missing_checkpoint_is_error(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root,True)
            (root/'toptee/checkpoints/toptee_structured_off_seed42_best.json').unlink()
            with self.assertRaises(FileNotFoundError):
                build(root)


if __name__=='__main__':
    unittest.main()
