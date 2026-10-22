"""Binding and durable-publication guards for the new full-category queue."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from generate_fashioniq_online import check_complete, fingerprint, VERSION
from run_fiq_shirt_toptee import publish_checkpoint


class OnlineGuards(unittest.TestCase):
    def payload(self):
        row = dict(source_index=0, category="shirt", split="val", candidate="ref",
                   target="target", modification_text="make the sleeves shorter")
        sample = dict(row, status="ok", validation={"valid": True}, semantic_validation={"valid": True},
                      generation_input={"target_was_provided_to_model": False})
        return [row], dict(state="complete", model="qwen3.7-flash", category="shirt", split="val",
                           query_sha256=fingerprint([row]), protocol={"version": VERSION}, samples=[sample])

    def test_rejects_stale_target_duplicate_and_target_input(self):
        rows, payload = self.payload()
        self.assertEqual(len(check_complete(payload, rows, "shirt", "val")), 1)
        for change in ("target", "duplicate", "leak"):
            altered = copy.deepcopy(payload)
            if change == "target":
                altered["samples"][0]["target"] = "other_target"
            elif change == "duplicate":
                altered["samples"] *= 2
            else:
                altered["samples"][0]["generation_input"]["target_was_provided_to_model"] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                check_complete(altered, rows, "shirt", "val")

    def test_low_coverage_blocks_training_without_removing_failed_queries(self):
        rows, payload = self.payload()
        payload["samples"][0]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "90%"):
            check_complete(payload, rows, "shirt", "val")
        self.assertEqual(len(payload["samples"]), 1)

    def test_publication_is_bit_identical_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / "staged.pt", root / "saved.pt"
            source.write_bytes(bytes(range(256)) * 129)
            source.with_suffix(".json").write_text(json.dumps({"epoch": 3}))
            with patch("run_fiq_shirt_toptee.require_space"):
                checksum = publish_checkpoint(source, destination)
                self.assertEqual(source.read_bytes(), destination.read_bytes())
                self.assertEqual(len(checksum), 64)
                with self.assertRaises(FileExistsError):
                    publish_checkpoint(source, destination)
            self.assertTrue(source.exists())


if __name__ == "__main__":
    unittest.main()
