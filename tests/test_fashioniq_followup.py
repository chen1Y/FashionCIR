import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from queue_fashioniq_augmentation import MODES, STAGES, merge_proposals, shoes_state


class FollowupTests(unittest.TestCase):
    def complete(self):
        return dict(status="complete", stages=[dict(stage=s, returncode=0) for s in STAGES],
                    metrics={m: dict(shoes_r1=30., shoes_r10=60., shoes_r50=80.) for m in MODES})

    def test_baseline_or_generation_does_not_unlock_followup(self):
        for stage in ("waiting_for_original_dqu", "qwen_train", "qwen_val", "structured_off", "structured_annotated"):
            self.assertEqual(shoes_state(dict(status=stage, stages=[])), "waiting")
        self.assertEqual(shoes_state(self.complete()), "ready")

    def test_stage_failure_blocks_followup(self):
        self.assertEqual(shoes_state(dict(status="stopped_on_error")), "failed")
        report = self.complete()
        report["stages"][0]["returncode"] = 1
        self.assertEqual(shoes_state(report), "failed")

    def test_incomplete_or_nonfinite_comparison_is_not_success(self):
        report = self.complete()
        report["stages"].pop()
        with self.assertRaises(ValueError):
            shoes_state(report)
        for value in (float("nan"), float("inf"), None, True, 101):
            report = self.complete()
            report["metrics"]["structured_off"]["shoes_r10"] = value
            with self.assertRaises(ValueError):
                shoes_state(report)

    def test_merge_keeps_pilot_and_limits_duplicate_target_degree(self):
        pilot = dict(category="dress", source_signature="bound", records=[
            dict(candidate="a", target="b", decision="accept"),
            dict(candidate="c", target="b", decision="accept")])
        proposals = dict(category="dress", source_signature="bound", records=[
            dict(candidate="a", target="b", decision="pending"),
            dict(candidate="d", target="b", decision="pending"),
            dict(candidate="e", target="f", decision="pending")])
        before = copy.deepcopy(pilot)
        merged = merge_proposals(pilot, proposals, 1)
        self.assertEqual(pilot, before)
        self.assertEqual(merged["records"][:2], pilot["records"])
        self.assertEqual(merged["records"][-1]["candidate"], "e")
        self.assertEqual(merged["new_attempt_budget"], 1)
        with self.assertRaises(ValueError):
            merge_proposals(pilot, proposals, 2)

    def test_stale_pilot_is_not_merged(self):
        with self.assertRaises(ValueError):
            merge_proposals(dict(source_signature="old", category="dress"),
                            dict(source_signature="new", category="dress"), 1)


if __name__ == "__main__":
    unittest.main()
