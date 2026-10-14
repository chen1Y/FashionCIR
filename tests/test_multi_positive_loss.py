import ast
import importlib
import random
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from multi_positive import PositiveIndex
from multi_positive_loss import multi_positive_nce, positive_aware_margin


class LossTest(unittest.TestCase):
    def assertTensorClose(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.allclose(actual, expected, rtol=1e-5, atol=1e-6),
                        msg=f"{actual.detach()} != {expected.detach()}")

    def test_diagonal_matches_cross_entropy_values_and_gradients(self):
        torch.manual_seed(7)
        logits = torch.randn(5, 5, requires_grad=True)
        expected = F.cross_entropy(logits, torch.arange(5))
        actual = multi_positive_nce(logits, torch.eye(5, dtype=torch.bool))
        self.assertTensorClose(actual, expected)
        self.assertTensorClose(torch.autograd.grad(actual, logits)[0],
                                   torch.autograd.grad(expected, logits)[0])
        self.assertTensorClose(multi_positive_nce(logits), expected)

    def test_false_negative_gradient_becomes_positive_attraction(self):
        logits = torch.tensor([[3., -1., 2.]], requires_grad=True)
        loss = multi_positive_nce(logits, torch.tensor([[True, True, False]]))
        gradient = torch.autograd.grad(loss, logits)[0]
        self.assertLess(gradient[0, 1], 0)
        self.assertGreater(gradient[0, 2], 0)
        expected = -F.log_softmax(logits, 1)[0, :2].mean()
        self.assertTensorClose(loss, expected)

    def test_rectangular_bank_and_column_permutation(self):
        logits = torch.randn(2, 4, requires_grad=True)
        mask = torch.tensor([[True, False, True, False], [False, True, False, False]])
        perm = torch.tensor([3, 0, 2, 1])
        self.assertTensorClose(multi_positive_nce(logits, mask),
                                   multi_positive_nce(logits[:, perm], mask[:, perm]))
        self.assertEqual(multi_positive_nce(logits, mask, "none").shape, (2,))

    def test_invalid_masks_are_rejected(self):
        for mask in [torch.zeros(2, 3, dtype=torch.bool), torch.ones(2, 2, dtype=torch.bool),
                     torch.ones(2, 3)]:
            with self.assertRaises(ValueError):
                multi_positive_nce(torch.randn(2, 3), mask)

    def test_large_half_precision_logits_have_finite_gradients(self):
        logits = torch.tensor([[10000., -10000., 0.]], dtype=torch.float16, requires_grad=True)
        loss = multi_positive_nce(logits, torch.tensor([[True, True, False]]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_hard_mining_excludes_off_diagonal_positives(self):
        baseline = torch.tensor([[.8, .99, .3]])
        adapted = torch.tensor([[.8, .2, .3]], requires_grad=True)
        mask = torch.tensor([[True, True, False]])
        loss, _ = positive_aware_margin(baseline, adapted, mask, torch.ones(1), 5, .01)
        loss.backward()
        self.assertLess(adapted.grad[0, 1], 0)  # positive, never repelled as a negative
        self.assertGreater(adapted.grad[0, 2], 0)

    def test_all_positive_or_no_valid_rows_are_safe(self):
        for mask, valid in [(torch.ones(2, 3, dtype=torch.bool), torch.ones(2)),
                            (torch.eye(2, 3, dtype=torch.bool), torch.zeros(2))]:
            adapted = torch.randn(2, 3, requires_grad=True)
            loss, rate = positive_aware_margin(torch.randn(2, 3), adapted, mask, valid, 8, .01)
            loss.backward()
            self.assertEqual(float(loss.detach()), 0)
            self.assertEqual(float(rate), 0)
            self.assertTrue(torch.isfinite(adapted.grad).all())

    def test_old_hard_negative_objective_is_preserved_when_disabled(self):
        torch.manual_seed(5)
        base, adapted = torch.randn(4, 4), torch.randn(4, 4, requires_grad=True)
        valid = torch.tensor([1., 1., 0., 1.])
        ids = base.masked_fill(torch.eye(4, dtype=torch.bool), -torch.inf).topk(2, 1).indices
        old = F.relu(.01 + base.diag()[:, None] - base.gather(1, ids)
                     - adapted.diag()[:, None] + adapted.gather(1, ids))
        expected = (old.mean(1) * valid).sum() / valid.sum()
        actual, _ = positive_aware_margin(base, adapted, None, valid, 2, .01)
        self.assertTensorClose(actual, expected)
        self.assertTensorClose(torch.autograd.grad(actual, adapted)[0],
                                   torch.autograd.grad(expected, adapted)[0])

    def test_compute_loss_with_rectangular_bank_backpropagates(self):
        # Real compute_loss, small synthetic features; no model download or CLIP mock outputs.
        with patch.dict(sys.modules, {"open_clip": types.ModuleType("open_clip")}):
            module = importlib.import_module("newModel")
        model = module.DQU_CIR.__new__(module.DQU_CIR)
        torch.nn.Module.__init__(model)
        model.loss_weight = torch.nn.Parameter(torch.tensor([10.]))
        model.residual = torch.nn.Parameter(torch.zeros(2, 3))
        model.gate = torch.nn.Parameter(torch.zeros(2, 1))
        base = F.normalize(torch.tensor([[1., 2., 3.], [3., 1., 2.]]), dim=1)
        def extract_query(self, *args, **kwargs):
            adapted = F.normalize(base + self.residual, dim=1)
            return adapted, dict(baseline_text=base, structured_text=base.flip(1),
                image_feature=base, adapted_text=adapted,
                predicted_gate_for_loss=self.gate.sigmoid(), mask_for_loss=torch.ones(2, 1),
                calibrated_confidence_for_loss=self.gate.sigmoid(),
                effective_residual_for_loss=self.residual,
                adapter_residual_for_loss=self.residual,
                structured_weight_for_loss=torch.ones(2, 1) * .1,
                dqu_text_weight=torch.ones(2, 1) * .5)
        model.extract_query = types.MethodType(extract_query, model)
        model.extract_target = lambda images: F.normalize(images, dim=1)
        for mask, targets in [(None, torch.eye(3)[:2]),
                              (torch.tensor([[True, False, True], [False, True, False]]), torch.eye(3)),
                              (torch.ones(2, 3, dtype=torch.bool), torch.eye(3))]:
            model.zero_grad()
            losses = model.compute_loss(["a", "b"], ["a", "b"], base, targets, positive_targets=mask)
            losses["loss"].backward()
            self.assertTrue(all(torch.isfinite(v).all() for v in losses.values()))
            self.assertTrue(torch.isfinite(model.residual.grad).all())
            self.assertTrue(torch.isfinite(model.gate.grad).all())

    def test_dataset_extra_preserves_query_and_padding(self):
        # Execute the real item-loading method without importing optional cv2/CLIP.
        tree = ast.parse((SRC / "newDataset.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FashionIQ")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__getitem__")
        namespace = {"random": random}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "newDataset.__getitem__", "exec"), namespace)
        q = dict(source_index=0, candidate="ref", target="a", modification_text="make blue")
        index = PositiveIndex([q], ["ref", "a", "b"], category="dress", mode="verified",
            manifest=dict(schema_version=1, split="train", category="dress",
                          queries=[dict(q, reviewed=True, positive_targets=["b"])]))
        fake = types.SimpleNamespace(train_data=[q], multi_positive_index=index, sample_extra_positive=True,
            _baseline_text=lambda *a: "original query",
            _structured_for=lambda *a: ("structured", ("field",) * 4, (True,) * 4, True, .9, (1.,) * 5),
            get_img=lambda image, stage: (torch.tensor([1. if image == "a" else 2.]), image),
            get_query_img=lambda *a, **kw: (torch.tensor([3.]), "ref"))
        item = namespace["__getitem__"](fake, 0)
        self.assertEqual(item["textual_query"], "original query")
        self.assertEqual(item["target"], "a")
        self.assertEqual(item["extra_target_id"], "b")
        self.assertTrue(item["has_extra_target"])
        index.positives[0] = {"a"}
        padded = namespace["__getitem__"](fake, 0)
        self.assertFalse(padded["has_extra_target"])
        batch = torch.utils.data.default_collate([item, padded])
        self.assertEqual(batch["extra_target_img"][batch["has_extra_target"]].shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
