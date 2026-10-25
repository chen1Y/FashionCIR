"""Prevent completed-baseline repetition and spending/runtime resets on migration."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from resume_fiq_saved_pause import remaining_commands, remaining_budget


class SavedPauseResume(unittest.TestCase):
    def test_completed_shirt_baseline_is_not_repeated(self):
        protocol = {"commands": {"shirt": [("dqu_original", ["baseline"]), ("structured_off", ["adapter"])]}}
        report = {"stages": [{"category": "shirt", "stage": "dqu_original", "status": "complete", "returncode": 0}]}
        plan = {"completed_stages": [["shirt", "dqu_original"]],
                "remaining_gpu_commands": [{"category": "shirt", "stage": "structured_off", "command": ["adapter"]}]}
        self.assertEqual(remaining_commands(protocol, report, plan), plan["remaining_gpu_commands"])
        changed = copy.deepcopy(plan)
        changed["remaining_gpu_commands"][0]["command"] = ["different_training_budget"]
        with self.assertRaises(ValueError):
            remaining_commands(protocol, report, changed)

    def test_failed_stage_cannot_be_adopted_as_complete(self):
        with self.assertRaises(ValueError):
            remaining_commands({"commands": {}},
                {"stages": [{"category": "shirt", "stage": "dqu_original", "status": "failed", "returncode": 1}]},
                {"completed_stages": [["shirt", "dqu_original"]], "remaining_gpu_commands": []})

    def test_user_pause_retains_consumed_runtime(self):
        audit = {"spent_wall_seconds_before_pause": 3152.034592,
                 "remaining_original_wall_budget_seconds": 40047.965408}
        self.assertAlmostEqual(remaining_budget(audit), 40047.965408)
        audit["remaining_original_wall_budget_seconds"] = 43200
        with self.assertRaises(ValueError):
            remaining_budget(audit)


if __name__ == "__main__":
    unittest.main()
