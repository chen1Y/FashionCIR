import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fiq_batch_repro_diagnostic_025 import fixed_batches,comparison


class FixedBatchDiagnosticTests(unittest.TestCase):
    def test_only_unique_metadata_conflict_and_fixed22_presentations(self):
        c=dict(image_id='b',old_batch_ids=['a','b','c'],new_batch_ids=['d','e','f','g','h','i','j','b'])
        image,batches=fixed_batches([c]);self.assertEqual(image,'b');self.assertEqual(sum(len(b['ids']) for b in batches),22);self.assertEqual(batches[0]['ids'],batches[2]['ids']);self.assertEqual(batches[3]['ids'][:3],c['old_batch_ids']);self.assertEqual(batches[3]['ids'].index(image),1)
        with self.assertRaises(ValueError):fixed_batches([])
        with self.assertRaises(ValueError):fixed_batches([c,c])
    def test_comparison_has_fixed_threshold_not_fitted_to_outputs(self):
        self.assertTrue(comparison([0.,1.],[0.,1.])['exact']);self.assertTrue(comparison([0.,1.],[1e-4,1.])['within_existing_1e4']);self.assertFalse(comparison([0.,1.],[.000101,1.])['within_existing_1e4'])
    def test_bad_geometry_or_nonfinite_vectors_rejected(self):
        with self.assertRaises(ValueError):fixed_batches([dict(image_id='b',old_batch_ids=['b'],new_batch_ids=['b']*8)])
        for a,b in (([],[]),([1],[1,2]),([float('nan')],[0]),([float('inf')],[0])):
            with self.assertRaises(ValueError):comparison(a,b)


if __name__=='__main__':unittest.main()
