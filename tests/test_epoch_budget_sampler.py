import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from epoch_budget_sampler import EpochBudgetSampler


class BudgetSamplerTests(unittest.TestCase):
    def test_all_unique_rows_and_common_prefix_slots(self):
        small, large = EpochBudgetSampler(5985, 6086, 42), EpochBudgetSampler(6086, 6086, 42)
        for epoch in (1, 2, 6):
            small.set_epoch(epoch)
            large.set_epoch(epoch)
            a, b = list(small), list(large)
            self.assertEqual(len(a), 6086)
            self.assertEqual(set(a), set(range(5985)))
            self.assertEqual(len(set(b)), 6086)
            self.assertTrue(all(x == y for x, y in zip(a, b) if y < 5985))
            self.assertEqual(sum(x != y for x, y in zip(a, b)), 101)
            self.assertEqual(a, list(small))

    def test_repeats_and_no_truncation(self):
        sampler = EpochBudgetSampler(3, 10, 7)
        self.assertEqual(len(list(sampler)), 10)
        self.assertEqual(set(sampler), {0, 1, 2})
        sampler.set_epoch(1)
        first = list(sampler)
        sampler.set_epoch(2)
        self.assertNotEqual(first, list(sampler))
        with self.assertRaises(ValueError):
            EpochBudgetSampler(5, 4, 42)


if __name__ == "__main__":
    unittest.main()
