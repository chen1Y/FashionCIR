from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shoes_online_pilot import Budget
from shoes_online_refine import conservative_prompt, inherited_budget, verify_parent_rows


class RefinementTests(unittest.TestCase):
    def test_cost_limit_is_shared_with_paid_parent(self):
        original = Budget(.5, 128)
        original.reserve(768)
        original.account({"prompt_tokens": 1000, "completion_tokens": 400})
        continued = inherited_budget(original.summary())
        continued.reserve(256)
        self.assertEqual(continued.calls, 2)
        self.assertGreater(continued.reserved_cny, original.reserved_cny)
        self.assertEqual(continued.estimated_cny, original.estimated_cny)
        continued.calls = 128
        with self.assertRaises(RuntimeError):
            continued.reserve(256)

    def test_unknown_cost_cannot_reset_budget(self):
        cost = Budget().summary()
        cost["usage_missing_calls"] = 1
        with self.assertRaises(ValueError):
            inherited_budget(cost)

    def test_global_color_and_ambiguous_placement_remain_explicit(self):
        prompt = conservative_prompt("have pink in them")
        self.assertIn("cannot retain a white toe box", prompt)
        self.assertIn("NOT that all existing red is removed", prompt)
        self.assertIn("DIFFERENT colored shoes", prompt)

    def test_non_contiguous_combined_source_indices_are_not_list_offsets(self):
        rows = [dict(source_index=i, candidate=f"r{i}", target=f"t{i}", modification_text="blue")
                for i in (58, 71, 236)]
        verify_parent_rows(rows, [dict(rows[1])])
        with self.assertRaises(ValueError):
            verify_parent_rows(rows, [dict(rows[1], target="wrong")])


if __name__ == "__main__":
    unittest.main()
