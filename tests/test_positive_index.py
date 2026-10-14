import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from multi_positive import PositiveIndex


class PositiveIndexTest(unittest.TestCase):
    def setUp(self):
        self.queries = [
            dict(source_index=0, candidate="ref", target="a", modification_text="make blue"),
            dict(source_index=1, candidate="ref", target="b", modification_text="make blue"),
            dict(source_index=2, candidate="other", target="a", modification_text="make blue"),
        ]
        self.ids = ["ref", "other", "a", "b", "c"]
        self.manifest = dict(schema_version=1, category="dress", split="train", queries=[
            dict(self.queries[0], reviewed=True, positive_targets=["c"])
        ])

    def build(self, mode="identity", manifest=None, queries=None):
        return PositiveIndex(queries if queries is not None else self.queries, self.ids,
                             category="dress", mode=mode, manifest=manifest)

    def test_identity_recognizes_repeated_target_ids(self):
        index = self.build()
        self.assertEqual(index.mask_rows([0, 1], ["a", "a", "b"]),
                         [[True, True, False], [False, False, True]])

    def test_exact_query_group_requires_same_reference(self):
        index = self.build("annotated")
        self.assertEqual(index.positives[0], {"a", "b"})
        self.assertEqual(index.positives[2], {"a"})
        self.assertEqual(index.extras(0), ["b"])

    def test_reviewed_additions_are_query_specific(self):
        index = self.build("verified", self.manifest)
        self.assertEqual(index.positives[0], {"a", "b", "c"})
        self.assertNotIn("c", index.positives[1])
        self.assertNotIn("c", index.positives[2])

    def test_reject_leakage_reference_unreviewed_and_stale_labels(self):
        changes = [dict(split="val"), dict(category="shirt"), dict(schema_version=2)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.build("verified", dict(self.manifest, **change))
        for change in [dict(reviewed=False), dict(positive_targets=["val-image"]),
                       dict(positive_targets=["ref"]), dict(candidate="other"),
                       dict(target="b"), dict(modification_text="make red"),
                       dict(source_index=99), dict(positive_targets="c")]:
            bad = copy.deepcopy(self.manifest)
            bad["queries"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.build("verified", bad)

    def test_duplicate_rows_and_missing_manifest(self):
        bad = copy.deepcopy(self.manifest)
        bad["queries"] *= 2
        with self.assertRaises(ValueError):
            self.build("verified", bad)
        with self.assertRaises(ValueError):
            self.build("verified")
        with self.assertRaises(ValueError):
            self.build(queries=self.queries * 2)

    def test_train_membership_and_reference_exclusion(self):
        for change in [dict(candidate="val-image"), dict(target="val-image"), dict(target="ref")]:
            queries = copy.deepcopy(self.queries)
            queries[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.build(queries=queries)

    def test_sparse_original_indices_survive_filtering(self):
        index = self.build(queries=[self.queries[2]])
        self.assertEqual(index.mask_rows([2], ["a", "b"]), [[True, False]])


if __name__ == "__main__":
    unittest.main()
