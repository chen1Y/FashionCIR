import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fashioniq_augmentation import build_overlay, context, propose, validate, write_json
from fashioniq_noise import CATEGORIES, file_sha
from generate_fashioniq_pair_captions import require_idle


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fashioncir-augmentation-test-")
        self.base = Path(self.temp.name).resolve()
        if self.base.parent != Path(tempfile.gettempdir()).resolve() or not self.base.name.startswith("fashioncir-augmentation-test-"):
            raise ValueError("Unexpected cleanup target")
        self.addCleanup(self.temp.cleanup)
        self.root = self.base / "original"
        for folder in ("captions", "image_splits", "resized_image"):
            (self.root / folder).mkdir(parents=True)
        for cat in CATEGORIES:
            for split, ids in (("train", ["ref", "blue", "green", "held"]), ("val", ["held"]), ("test", ["test"])):
                write_json(self.root / f"image_splits/split.{cat}.{split}.json", ids)
        self.original = [dict(candidate="ref", target="green", captions=["make it green", "keep the long skirt"])]
        write_json(self.root / "captions/cap.dress.train.json", self.original)
        write_json(self.root / "captions/cap.dress.val.json", [dict(candidate="held", target="held", captions=["same", "same"])])
        write_json(self.root / "captions/keywords_in_mods_dress.json", {"ref_green": ["green"]})
        write_json(self.root / "captions/image_captions_dress_train.json", {
            "ref": "a woman in a red sleeveless long dress", "blue": "a woman in a blue sleeveless long dress",
            "green": "a woman in a green sleeveless long dress", "held": "a woman in a black sleeveless long dress"})
        for i, color in (("ref", "red"), ("blue", "blue"), ("green", "green"), ("held", "black")):
            Image.new("RGB", (40, 40), color).save(self.root / f"resized_image/{i}.jpg")
        self.manifest = propose(self.root, "dress", limit=1)
        row = self.manifest["records"][0]
        # Explicit novel pair for assertions independent of candidate ranking.
        row.update(candidate="ref", target="blue", reference_sha256=file_sha(self.root / "resized_image/ref.jpg"),
                   target_sha256=file_sha(self.root / "resized_image/blue.jpg"),
                   captions=["change the red dress to blue", "keep the sleeveless long style"],
                   target_keywords=["blue", "sleeveless"], decision="accept",
                   review=dict(reviewer="test fixture", both_garments=True, caption_satisfied=[True, True],
                               meaningful_difference=True, target_keywords_supported=True, evidence="fixture"))

    def test_candidates_exclude_heldout_and_existing_pairs(self):
        proposals = propose(self.root, "dress", limit=10)
        self.assertEqual(proposals["global_heldout_overlap_excluded"], 1)
        for r in proposals["records"]:
            self.assertNotIn("held", (r["candidate"], r["target"]))
            self.assertNotEqual((r["candidate"], r["target"]), ("ref", "green"))
            self.assertEqual(r["decision"], "pending")
        self.assertEqual(proposals, propose(self.root, "dress", limit=10))

    def test_train_leak_existing_identity_and_duplicate_additions_rejected(self):
        for ref, target in (("ref", "held"), ("ref", "green"), ("ref", "ref")):
            m = copy.deepcopy(self.manifest)
            m["records"][0].update(candidate=ref, target=target)
            with self.assertRaises(ValueError):
                validate(self.root, m)
        m = copy.deepcopy(self.manifest)
        m["records"] *= 2
        with self.assertRaises(ValueError):
            validate(self.root, m)

    def test_stale_metadata_or_changed_image_rejected(self):
        row = self.manifest["records"][0]
        Image.new("RGB", (40, 40), "orange").save(self.root / f"resized_image/{row['target']}.jpg")
        with self.assertRaisesRegex(ValueError, "changed image"):
            validate(self.root, self.manifest)
        write_json(self.root / "captions/unrelated_metadata.json", {})
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            validate(self.root, self.manifest)

    def test_identical_photo_different_id_rejected(self):
        (self.root / "resized_image/blue.jpg").write_bytes((self.root / "resized_image/ref.jpg").read_bytes())
        self.manifest["records"][0]["target_sha256"] = file_sha(self.root / "resized_image/blue.jpg")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate(self.root, self.manifest)

    def test_uncertain_or_incomplete_labels_not_accepted(self):
        for change in ({"captions": ["left photo", "second image"]}, {"target_keywords": []},
                       {"review": {}}, {"review": dict(self.manifest["records"][0]["review"], caption_satisfied=[True, None])}):
            m = copy.deepcopy(self.manifest)
            m["records"][0].update(change)
            with self.assertRaises(ValueError):
                validate(self.root, m)

    def test_overlay_preserves_original_prefix_and_images_and_heldout(self):
        before = context(self.root, "dress")["source_signature"]
        out = self.base / "overlay"
        try:
            summary = build_overlay(self.root, self.manifest, out)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                self.skipTest("Windows symlink privilege unavailable; verified on Linux server")
            raise
        rows = json.loads((out / "captions/cap.dress.train.json").read_text())
        self.assertEqual(rows[:-1], self.original)
        self.assertEqual(summary["added_physical_images"], 0)
        self.assertEqual((out / "resized_image").resolve(), (self.root / "resized_image").resolve())
        self.assertEqual((out / "image_splits").resolve(), (self.root / "image_splits").resolve())
        self.assertEqual(file_sha(out / "captions/cap.dress.val.json"), file_sha(self.root / "captions/cap.dress.val.json"))
        keywords = json.loads((out / "captions/keywords_in_mods_dress.json").read_text())
        self.assertEqual(keywords["ref_blue"], ["blue, sleeveless"])
        self.assertEqual(context(self.root, "dress")["source_signature"], before)
        with self.assertRaises(FileExistsError):
            build_overlay(self.root, self.manifest, out)

    def test_original_directory_cannot_be_used_as_output(self):
        with self.assertRaises(ValueError):
            build_overlay(self.root, self.manifest, self.root / "overlay")

    def test_budget_and_degree_caps(self):
        with self.assertRaises(ValueError):
            validate(self.root, self.manifest, max_fraction=0)

    def test_labeler_refuses_live_queue_before_touching_gpu(self):
        guard = self.base / "guard"
        guard.mkdir()
        (guard / "queue.pid").write_text("1234")
        with patch("generate_fashioniq_pair_captions.os.kill"), patch("generate_fashioniq_pair_captions.subprocess.check_output") as gpu:
            with self.assertRaisesRegex(RuntimeError, "benchmark PID"):
                require_idle(guard, 8, self.base / "labels.json")
            gpu.assert_not_called()

    def test_labeler_refuses_occupied_gpu(self):
        with patch("generate_fashioniq_pair_captions.subprocess.check_output", return_value="999\n"):
            with self.assertRaisesRegex(RuntimeError, "compute processes"):
                require_idle(self.base / "absent", 8, self.base / "labels.json")


if __name__ == "__main__":
    unittest.main()
