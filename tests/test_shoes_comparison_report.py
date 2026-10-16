import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("shoes_report", ROOT / "scripts/report_shoes_comparison.py")
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


class ComparisonReportTests(unittest.TestCase):
    def setUp(self):
        directory = Path(tempfile.gettempdir()) if os.name == "posix" else ROOT / ".autodl/test_tmp"
        directory.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="fashioncir-comparison-test-", dir=directory)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.report = dict(config=dict(dataset="shoes", seed=42), status="qwen_train", stages=[],
                           structured_paths={"train": str(self.root / "train.json"), "val": str(self.root / "val.json")})
        self.values = dict(shoes_r1=30., shoes_r10=68., shoes_r50=88., epoch=3)
        self.put("experiment_report.json", self.report)
        self.put("checkpoints/shoes_dqu_original_seed42_best.json", self.values)

    def put(self, path, data):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def test_pending_is_not_zero_or_final_and_paper_has_separate_uncertainty(self):
        data = reporter.build(self.root)
        self.assertFalse(data["final_comparison_complete"])
        self.assertIsNone(data["rows"][2]["metrics"])
        self.assertEqual(data["rows"][0]["std"]["shoes_r1"], 1.31)
        self.assertEqual(data["rows"][0]["random_seeds"], 5)
        self.assertIn("—", reporter.markdown(data))

    def test_existing_checkpoint_during_training_is_intermediate(self):
        self.put("checkpoints/shoes_structured_off_seed42_best.json", dict(self.values, shoes_r1=31.))
        data = reporter.build(self.root)
        self.assertEqual(data["rows"][2]["status"], "intermediate_best")
        self.assertEqual(data["rows"][2]["delta_vs_local_baseline_pp"]["shoes_r1"], 1.)
        self.assertFalse(data["final_comparison_complete"])

    def test_missing_or_nonfinite_metric_is_rejected(self):
        for values in (dict(shoes_r1=1, shoes_r10=2), dict(self.values, shoes_r50=math.nan), dict(self.values, shoes_r1=True)):
            with self.assertRaises((KeyError, ValueError)):
                reporter.metrics(values)

    def test_final_requires_all_successful_stages_and_all_checkpoints(self):
        self.report["status"] = "complete"
        names = ["qwen_train", "qwen_val", *(f"structured_{m}" for m in reporter.MODES)]
        self.report["stages"] = [dict(stage=n, returncode=0) for n in names]
        self.put("experiment_report.json", self.report)
        with self.assertRaises(FileNotFoundError):
            reporter.build(self.root)
        for mode in reporter.MODES:
            self.put(f"checkpoints/shoes_structured_{mode}_seed42_best.json", self.values)
        self.assertTrue(reporter.build(self.root)["final_comparison_complete"])
        self.report["stages"][0]["returncode"] = 1
        self.put("experiment_report.json", self.report)
        self.assertFalse(reporter.build(self.root)["final_comparison_complete"])


if __name__ == "__main__":
    unittest.main()
