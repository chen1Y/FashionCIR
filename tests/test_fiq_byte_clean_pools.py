from pathlib import Path
import importlib.util
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from fiq_candidate_partition import image_bucket
from fiq_byte_clean_pools import clean_partition, positive_byte_sets, verify_actual_inputs

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from fiq_byte_clean_pools import clean_training_pool
    from fiq_candidate_head import candidate_features, fit_statistics, CandidateHead


class BytePartitionTests(unittest.TestCase):
    def setUp(self):
        self.ids = [f'image{i}' for i in range(600)]
        self.hashes = {i: 'sha_'+i for i in self.ids}
        self.fit_ids = [i for i in self.ids if image_bucket(i) != 0]
        self.dev_ids = [i for i in self.ids if image_bucket(i) == 0]

    def row(self, a, b):
        return dict(candidate=a, target=b, modification_text='make it blue')

    def test_dev_priority_and_heldout_filter_both_ref_and_target(self):
        f, d, h = self.fit_ids, self.dev_ids, self.hashes
        h[f[0]] = h[d[0]]
        h[f[3]] = 'official'
        h[d[3]] = 'official'
        fit = [self.row(f[0], f[1]), self.row(f[2], f[3]), self.row(f[4], f[5])]
        dev = [self.row(d[2], d[3]), self.row(d[4], d[5])]
        p = clean_partition(self.ids, h, fit, dev, {'official'})
        self.assertEqual(p['masks']['fit'], [False, False, True])
        self.assertEqual(p['masks']['dev'], [False, True])
        self.assertNotIn(self.ids.index(f[0]), p['banks']['fit'])
        self.assertEqual(p['reason_counts']['fit']['reference_dev_bank_byte'], 1)
        fit_sha = {h[self.ids[i]] for i in p['banks']['fit']}
        dev_sha = {h[self.ids[i]] for i in p['banks']['dev']}
        self.assertFalse(fit_sha & dev_sha)
        self.assertNotIn('official', fit_sha | dev_sha)

    def test_positive_actual_id_not_required_to_be_negative_representative(self):
        a, b = self.fit_ids[:2]
        self.hashes[b] = self.hashes[a]
        p = clean_partition(self.ids, self.hashes, [], [], set())
        self.assertIn(self.ids.index(b), p['banks']['fit'])
        self.assertNotIn(self.ids.index(b), p['canonical']['fit'])

    def test_known_positives_link_reference_byte_aliases(self):
        f = self.fit_ids
        self.hashes[f[1]] = self.hashes[f[0]]
        known = positive_byte_sets([self.row(f[0], f[2]), self.row(f[1], f[3])], self.hashes)
        self.assertEqual(known[(self.hashes[f[0]], 'make it blue')], {self.hashes[f[2]], self.hashes[f[3]]})

    def test_actual_proof_rejects_positive_alias_negative(self):
        f = self.fit_ids
        row = self.row(f[0], f[1])
        pool = [self.ids.index(i) for i in f[1:33]]
        self.hashes[f[2]] = self.hashes[f[1]]
        with self.assertRaisesRegex(ValueError, 'Byte alias'):
            verify_actual_inputs(self.ids, self.hashes, {'fit': [row], 'dev': []}, {'fit': [0], 'dev': []},
                                 {'fit': [pool], 'dev': []}, {'fit': list(range(len(self.ids))), 'dev': []}, set())

    def test_missing_sha_is_hard_failure(self):
        del self.hashes[self.ids[-1]]
        with self.assertRaises(ValueError):
            clean_partition(self.ids, self.hashes, [], [], set())


@unittest.skipUnless(TORCH, 'Torch checks run on server')
class BytePoolTorchTests(unittest.TestCase):
    setUp = BytePartitionTests.setUp
    row = BytePartitionTests.row

    def test_noncanonical_gt_and_31_unique_negatives(self):
        f = self.fit_ids
        self.hashes[f[2]] = self.hashes[f[1]]
        rows = [self.row(f[0], f[2])]
        p = clean_partition(self.ids, self.hashes, rows, [], set())
        torch.manual_seed(42)
        gallery = torch.nn.functional.normalize(torch.randn(len(self.ids), 8), dim=-1)
        q = gallery[[self.ids.index(f[1])]]
        pool, base = clean_training_pool(q, q, gallery, torch.tensor([self.ids.index(f[0])]),
            torch.tensor([self.ids.index(f[2])]), self.ids, self.hashes, rows, [0], p['banks']['fit'], p['canonical']['fit'])
        self.assertEqual(int(pool[0, 0]), self.ids.index(f[2]))
        self.assertEqual(len({self.hashes[self.ids[i]] for i in pool[0].tolist()}), 32)
        self.assertTrue(torch.isfinite(base).all())
        verify_actual_inputs(self.ids, self.hashes, {'fit': rows, 'dev': []}, {'fit': [0], 'dev': []},
            {'fit': pool.tolist(), 'dev': []}, p['banks'], set())

    def test_insufficient_byte_unique_negatives_stops(self):
        f = self.fit_ids
        rows = [self.row(f[0], f[1])]
        gallery = torch.randn(len(self.ids), 8)
        with self.assertRaisesRegex(ValueError, 'Fewer than31'):
            clean_training_pool(gallery[:1], gallery[:1], gallery, torch.tensor([self.ids.index(f[0])]),
                torch.tensor([self.ids.index(f[1])]), self.ids, self.hashes, rows, [0],
                [self.ids.index(i) for i in f[:3]], [self.ids.index(i) for i in f[:3]])

    def test_blocked_rows_do_not_influence_statistics_or_gradients(self):
        x = torch.randn(4, 32, 6)
        active = [1, 3]
        x[[0, 2]] = 1e9
        mean, std = fit_statistics(x[active])
        self.assertLess(float(mean.abs().max()), 1.)
        head = CandidateHead(mean, std)
        value = head(x[active], torch.ones(2, dtype=torch.bool)).square().sum()
        value.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters()))


if __name__ == '__main__':
    unittest.main()
