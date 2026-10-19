"""Exercise real Diffusers attention and native CLIP token replacement on CPU."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from diffusers import DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextConfig, CLIPTextModel
from cig_trial import (TrialBudget, diffusion_loss, file_sha, mask_keyword, replace_pseudo_embeddings,
                       trainable_kv)


class TrialTests(unittest.TestCase):
    def test_real_unet_updates_only_cross_attention_kv(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
              layers_per_block=1, block_out_channels=(32, 32),
              down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
              up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=32,
              attention_head_dim=4)
        selected = trainable_kv(model)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer = torch.optim.AdamW(selected.values(), lr=1e-3)
        generator = torch.Generator().manual_seed(10)
        loss = diffusion_loss(model, DDPMScheduler(num_train_timesteps=20),
                              torch.randn(2, 4, 8, 8), torch.randn(2, 8, 32), generator)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in selected.values()))
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters() if name not in selected))
        optimizer.step()
        self.assertTrue(any(not torch.equal(p, before[name]) for name, p in selected.items()))
        self.assertTrue(all(torch.equal(p, before[name]) for name, p in model.named_parameters() if name not in selected))

    def test_native_clip_replacement_and_hook_removal(self):
        model = CLIPTextModel(CLIPTextConfig(vocab_size=20, hidden_size=32, intermediate_size=64,
                    projection_dim=32, num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=8))
        ids = torch.tensor([[1, 5, 8, 2, 2, 2, 2, 2]])
        baseline = model(ids).last_hidden_state.detach()
        pseudo = torch.randn(1, 32, requires_grad=True)
        def hook(module, inputs, output):
            return replace_pseudo_embeddings(ids, output, pseudo, 5)
        handle = model.text_model.embeddings.token_embedding.register_forward_hook(hook)
        output = model(ids).last_hidden_state
        output[0, 2].square().sum().backward()
        handle.remove()
        self.assertIsNotNone(pseudo.grad)
        self.assertFalse(torch.equal(output, baseline))
        self.assertTrue(torch.equal(model(ids).last_hidden_state.detach(), baseline))
        with self.assertRaises(ValueError):
            replace_pseudo_embeddings(torch.tensor([[5, 5]]), torch.zeros(1, 2, 32), pseudo, 5)

    def test_mask_and_budget_cannot_reset(self):
        class Tagger:
            def tag(self, words):
                return list(zip(words, ["DT", "JJ", "NN", "IN", "NNS"]))
        self.assertEqual(mask_keyword("a red dress with sleeves", Tagger()), ("with sleeves", "red dress"))
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data.bin"
            data.write_bytes(b"abc")
            self.assertEqual(file_sha(data), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
            path = Path(directory) / "budget.json"
            first = TrialBudget(path, 60)
            second = TrialBudget(path, 60)
            self.assertEqual(first.state, second.state)
            with self.assertRaises(ValueError):
                TrialBudget(path, 120)
            with self.assertRaises(TimeoutError):
                second.check(61)

    def test_nonfinite_loss_is_rejected(self):
        from types import SimpleNamespace
        class Broken(torch.nn.Module):
            def forward(self, noisy, timesteps, encoder_hidden_states):
                return SimpleNamespace(sample=torch.full_like(noisy, float("nan")))
        with self.assertRaises(FloatingPointError):
            diffusion_loss(Broken(), DDPMScheduler(num_train_timesteps=20),
                           torch.zeros(1, 4, 8, 8), torch.zeros(1, 8, 32), torch.Generator())


if __name__ == "__main__":
    unittest.main()
