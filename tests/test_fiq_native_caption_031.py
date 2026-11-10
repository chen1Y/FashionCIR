import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fiq_native_caption_math_031 import native_queries, metrics, transitions


class NativeCaptionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(source_index=0, candidate='ref', target='gt', modification_text='raw unchanged')]
        self.original = [dict(candidate='ref', target='gt', captions=['Longer-Sleevs!', 'LIGHTER.'])]

    def test_exact_native_correction_and_caption_separator(self):
        result = native_queries(self.rows, self.original, {'ref': 'A black tank.'}, {'sleevs': 'sleeves'})
        self.assertEqual(result[0]['native_text'], 'A black tank., but longer sleeves and lighter')

    def test_never_rewrite_raw_or_attributes(self):
        saved = copy.deepcopy((self.rows, self.original))
        native_queries(self.rows, self.original, {'ref': 'a tank'}, {})
        self.assertEqual((self.rows, self.original), saved)

    def test_wrong_source_identity_rejected(self):
        self.rows[0]['target'] = 'wrong'
        with self.assertRaises(ValueError):
            native_queries(self.rows, self.original, {'ref': 'a tank'}, {})

    def test_missing_caption_and_bad_index_rejected(self):
        with self.assertRaises(KeyError):
            native_queries(self.rows, self.original, {}, {})
        with self.assertRaises(ValueError):
            native_queries(self.rows, self.original, {'ref': ''}, {})
        self.rows[0]['source_index'] = -1
        with self.assertRaises(ValueError):
            native_queries(self.rows, self.original, {'ref': 'a tank'}, {})

    def test_four_way_transitions_recompute_metric_difference(self):
        after, before = [1, 11, 10, 60], [2, 10, 11, 60]
        t = transitions(after, before, 10)
        self.assertEqual([t[x] for x in ('gains', 'losses', 'both', 'neither')], [1, 1, 1, 1])
        self.assertEqual(t['net_pp'], metrics(after)['r10'] - metrics(before)['r10'])


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Server Torch required')
class NativeFusionTests(unittest.TestCase):
    def test_caption_swap_preserves_slots_and_missing_slots_fall_back(self):
        from types import SimpleNamespace
        from fiq_fusion_029 import DQUFusionHead, TextMixtureGate, fuse_parts
        from fiq_gate_patch_model_028 import AttributeGate
        normal = lambda t: torch.nn.functional.normalize(t, dim=-1)
        model = SimpleNamespace(attribute_gate=AttributeGate(), text_mixture=TextMixtureGate(),
                                image_fusion=DQUFusionHead(8, 0.))
        model.image_fusion.eval()
        ref, written, caption = [normal(torch.randn(2, 8)) for _ in range(3)]
        parts = dict(mod=normal(torch.randn(2, 8)), eos=normal(torch.randn(2, 8)),
                     slots=normal(torch.randn(2, 4, 8)), valid=torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.bool))
        saved = {k: v.clone() for k, v in parts.items()}
        raw = fuse_parts(model, parts, ref, written, 'full_learned_written')
        new = fuse_parts(model, dict(parts, mod=caption), ref, written, 'full_learned_written')
        self.assertTrue(torch.equal(raw['slot_weights'], new['slot_weights']))
        fallback, _ = model.image_fusion(caption, written)
        self.assertEqual(float(new['beta'][1]), 0.)
        self.assertTrue(torch.equal(new['q'][1], fallback[1]))
        self.assertTrue(all(torch.equal(parts[k], saved[k]) for k in parts))

    def test_beta_zero_with_native_text_and_native_head_is_native_query(self):
        from fiq_fusion_029 import DQUFusionHead
        normal = lambda t: torch.nn.functional.normalize(t, dim=-1)
        head = DQUFusionHead(8, 0.)
        head.eval()
        native, written = [normal(torch.randn(3, 8)) for _ in range(2)]
        expected, _ = head(native, written)
        composed = normal((1 - torch.zeros(3, 1)) * native)
        actual, _ = head(composed, written)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
