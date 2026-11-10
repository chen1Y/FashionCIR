import unittest

import torch
from torch.nn import functional as F

from fiq_fusion_029 import objective
from fiq_pixel_free_038 import PixelFreeVisual, VisualConflictEraserV2, visual_losses


class PixelFreeVisualTests(unittest.TestCase):
    def test_official_report_retains_ranks_and_paired_counts(self):
        from run_fiq_pixel_free_038 import official_result

        result = official_result([5, 11, 50], [15, 9, 55])
        self.assertEqual(result['ranks'], [5, 11, 50])
        self.assertEqual(result['vs_037']['10']['gains'], 1)
        self.assertEqual(result['vs_037']['10']['losses'], 1)
        self.assertEqual(result['r50'], 100.)

    def test_zero_start_recovers_reference(self):
        visual = PixelFreeVisual(8, True, True).eval()
        with torch.no_grad():
            visual.eraser.gamma.zero_()
        reference = F.normalize(torch.randn(3, 8), dim=1)
        text = F.normalize(torch.randn_like(reference), dim=1)
        drop = F.normalize(torch.randn_like(reference), dim=1)
        final, proxy = visual(reference, text, drop, torch.tensor([True, False, True]))
        torch.testing.assert_close(final, reference, atol=1e-6, rtol=0)
        torch.testing.assert_close(proxy, reference, atol=1e-6, rtol=0)

    def test_empty_drop_finite_backward_and_detached_target(self):
        visual = PixelFreeVisual(8, True, True)
        reference = F.normalize(torch.randn(3, 8), dim=1)
        text = F.normalize(torch.randn_like(reference), dim=1).requires_grad_()
        target = F.normalize(torch.randn_like(reference), dim=1).requires_grad_()
        final, proxy = visual(reference, text, torch.zeros_like(reference), torch.zeros(3, dtype=torch.bool))
        rec, kl = visual_losses(proxy, final, target)
        (rec + kl).backward()
        self.assertTrue(torch.isfinite(text.grad).all())
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in visual.parameters()))
        self.assertIsNone(target.grad)

    def test_eraser_does_not_remove_negative_or_missing_direction(self):
        eraser = VisualConflictEraserV2(initial_gamma=1.)
        reference = F.normalize(torch.tensor([[1., 1.], [-1., 1.], [1., 0.], [1., 1.]]), dim=-1)
        drop = torch.tensor([[1., 0.]]).expand_as(reference)
        clean = eraser(reference, drop, torch.tensor([True, True, True, False]))
        torch.testing.assert_close(clean[0], torch.tensor([0., 1.]))
        torch.testing.assert_close(clean[1:], reference[1:])

    def test_reference_weight_adds_once_to_masked_denominator(self):
        query = torch.tensor([[1., 0.]])
        gallery = torch.tensor([[0., 1.], [1., 0.], [-1., 0.]])
        mask = torch.tensor([[True, False, True]])
        out = dict(q=query, global_text=gallery[:1])
        loss = objective(out, gallery, torch.tensor([0]), mask, None, 'original_written',
                         reference=gallery[1:2], reference_weight=.2)
        expected = (1 + torch.exp(torch.tensor(-1 / .07)) + .2 * torch.exp(torch.tensor(1 / .07))).log()
        torch.testing.assert_close(loss['retrieval'], expected)
