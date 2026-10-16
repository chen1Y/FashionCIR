import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate_benchmark_edits import attempt_progress, resume_samples


class BenchmarkResumeTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(source_index=i, candidate=f"ref{i}", target=f"target{i}", modification_text="lighter") for i in (0, 7, 20)]
        self.saved = [dict(self.rows[0], query_index=0, status="ok", structured_edit={"evidence": "keep"}),
                      dict(self.rows[1], query_index=7, status="failed", error="original rejection")]

    def test_saved_successes_and_rejections_are_preserved_and_skipped(self):
        original = copy.deepcopy(self.saved)
        stored = resume_samples(self.rows, self.saved)
        remaining = [r["source_index"] for r in self.rows if r["source_index"] not in stored]
        self.assertEqual(remaining, [20])
        self.assertEqual(self.saved, original)
        self.assertIs(stored[7], self.saved[1])
        self.assertEqual(attempt_progress(self.rows, stored), dict(attempted_queries=2, total_queries=3,
                         remaining_queries=1, attempts_complete=False, valid_queries=1))

    def test_all_attempted_does_not_mean_all_valid(self):
        stored = resume_samples(self.rows[:2], self.saved)
        progress = attempt_progress(self.rows[:2], stored)
        self.assertTrue(progress["attempts_complete"])
        self.assertEqual(progress["valid_queries"], 1)
        self.assertEqual(progress["remaining_queries"], 0)

    def test_duplicate_foreign_or_changed_saved_samples_rejected(self):
        cases = [self.saved + [self.saved[0]]]
        for changes in (dict(query_index=99), dict(source_index=99), dict(target="wrong"), dict(status="pending")):
            s = copy.deepcopy(self.saved)
            s[0].update(changes)
            cases.append(s)
        for samples in cases:
            with self.assertRaises(ValueError):
                resume_samples(self.rows, samples)


if __name__ == "__main__":
    unittest.main()
