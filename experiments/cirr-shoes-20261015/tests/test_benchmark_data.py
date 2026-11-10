import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from benchmark_data import BenchmarkData, safe_path, selection_score


class BenchmarkDataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fashioncir-benchmark-test-")
        self.root = Path(self.temp.name).resolve()
        if self.root.parent != Path(tempfile.gettempdir()).resolve() or not self.root.name.startswith("fashioncir-benchmark-test-"):
            raise ValueError("Unexpected test cleanup target")
        self.addCleanup(self.temp.cleanup)

    def put(self, name, payload):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload))

    def cirr_fixture(self, nested=False):
        prefix = "captions/" if nested else ""
        for split, folder in (("train", "train/0"), ("val", "dev")):
            ids = [f"{split}{i}" for i in range(6)]
            mapping = {name: f"./{folder}/{name}.png" for name in ids}
            self.put(prefix+f"image_splits/split.rc2.{split}.json", mapping)
            self.put(prefix+f"captions/cap.rc2.{split}.json", [dict(reference=ids[0], target_hard=ids[1],
                caption="one dog", pairid=10, img_set=dict(members=ids))])
            self.put(f"image_captions_cirr_{split}.json", {name:"two dogs" for name in ids})
            self.put(f"keywords_in_mods_cirr_{split}.json", {ids[0]+"+"+ids[1]: ["one dog"]})

    def test_cirr_flat_and_original_layout_without_test1(self):
        self.cirr_fixture()
        x = BenchmarkData(self.root, "cirr")
        self.assertEqual(x.audit()["missing_images"], 12)
        self.assertFalse(x.audit()["test1_loaded"])
        self.assertEqual(x.baseline_text(x.queries["val"][0], "val"), "two dogs, butone dog")
        self.assertEqual(x.image_paths["train0"], self.root / "images/train/0/train0.png")

    def test_nested_cirr_and_source_reference_contract(self):
        self.cirr_fixture(nested=True)
        x = BenchmarkData(self.root, "cirr")
        self.assertEqual(x.queries["val"][0]["subset_id"], [f"val{i}" for i in range(6)])

    def test_cirr_split_leak_and_subset_validation(self):
        self.cirr_fixture()
        p = self.root / "captions/cap.rc2.val.json"
        rows = json.loads(p.read_text()); rows[0]["img_set"]["members"][-1] = "train0"
        p.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError, "subset"):
            BenchmarkData(self.root, "cirr")

    def test_reject_path_escape(self):
        with self.assertRaises(ValueError):
            safe_path(self.root, "../outside.png")

    def test_selection_uses_all_seven_recalls_not_diagnostics(self):
        m = {f"cirr_r{k}": float(k) for k in (1, 5, 10, 50)}
        m.update({f"cirr_subset_r{k}": float(k) for k in (1, 2, 3)})
        m["cirr_mean_rank"] = 10000
        self.assertEqual(selection_score(m, "cirr"), 72)
        self.assertEqual(selection_score({"shoes_r10": 20, "shoes_r50": 40}, "shoes"), 60)

    def test_cirr_subset_rank_excludes_source_and_uses_gallery_order(self):
        import torch
        from benchmark_metrics import cirr_metrics
        gallery = {name: i for i, name in enumerate(["x", "s", "a", "b", "t", "c", "d"])}
        scores = torch.tensor([[.99, 1., .2, .3, .8, .1, .0]])
        row = dict(source_img_id="s", target_img_id="t", subset_id=["s", "d", "b", "a", "t", "c"])
        out = dict(cirr_metrics(scores, [row], gallery, [2]))
        self.assertEqual(out["cirr_subset_r1"], 100)
        self.assertEqual(out["cirr_r5"], 100)

    def test_shoes_self_target_is_explicit_and_never_added_by_manifest(self):
        from multi_positive import PositiveIndex
        q = dict(source_index=0, candidate="a", target="a", modification_text="identical")
        with self.assertRaises(ValueError):
            PositiveIndex([q], ["a"], category="shoes")
        self.assertEqual(PositiveIndex([q], ["a"], category="shoes", allow_self_target=True).positives[0], {"a"})
        q2 = dict(q, source_index=1, target="b")
        x = PositiveIndex([q, q2], ["a", "b"], category="shoes", mode="annotated", allow_self_target=True)
        self.assertNotIn("a", x.positives[1])


if __name__ == "__main__":
    unittest.main()
