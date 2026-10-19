"""Small SD1.5 CIG-inspired domain trial, independent of the DQU training code.

Self reconstruction with frozen SEARLE/CLIP/VAE and cross-attention K/V updates.
This is neither an author checkpoint nor a reproduction of their CC3M experiment.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import torch
from torch import nn


class Phi(nn.Module):
    # Architecture used by SEARLE (ICCV 2023), https://github.com/miccunifi/SEARLE.
    def __init__(self, dim=768):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(0),
                                    nn.Linear(dim * 4, dim * 4), nn.GELU(), nn.Dropout(0),
                                    nn.Linear(dim * 4, dim))

    def forward(self, value):
        return self.layers(value)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def mask_keyword(caption, tagger):
    """Remove the first adjective* + noun+ span; record the exact removed words."""
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[^\w\s]", caption)
    tagged = tagger.tag(words)
    for start in range(len(words)):
        end = start
        while end < len(words) and tagged[end][1].startswith("JJ"):
            end += 1
        noun_start = end
        while end < len(words) and tagged[end][1].startswith("NN"):
            end += 1
        if end > noun_start:
            result = " ".join(words[:start] + words[end:])
            # Avoid an orphaned initial determiner after the subject is removed.
            result = re.sub(r"^(a|an|the)\s+", "", result, flags=re.I).strip(" ,.")
            return result, " ".join(words[start:end])
    raise ValueError("Caption has no noun phrase to mask")


def replace_pseudo_embeddings(ids, embeddings, pseudo, placeholder):
    matches = ids.eq(placeholder)
    if not torch.all(matches.sum(1) == 1):
        raise ValueError("Each prompt must contain exactly one nontruncated placeholder")
    if embeddings.shape[0] != pseudo.shape[0] or embeddings.shape[-1] != pseudo.shape[-1]:
        raise ValueError("Pseudo-token dimensions do not match CLIP token embeddings")
    return torch.where(matches.unsqueeze(-1), pseudo[:, None].to(embeddings.dtype), embeddings)


def encode_prompts(tokenizer, encoder, pseudo, captions):
    """Use the installed CLIP native causal/EOS logic, replacing only token inputs."""
    placeholder = tokenizer("$", add_special_tokens=False).input_ids
    if len(placeholder) != 1:
        raise ValueError("Tokenizer does not map $ to a single token")
    texts = ["a photo of $ that " + text for text in captions]
    lengths = tokenizer(texts, truncation=False).input_ids
    if any(len(row) > tokenizer.model_max_length for row in lengths):
        raise ValueError("Prompt exceeds CLIP context; truncation is forbidden")
    tokens = tokenizer(texts, padding="max_length", max_length=tokenizer.model_max_length,
                       truncation=False, return_tensors="pt")
    device = next(encoder.parameters()).device
    ids = tokens.input_ids.to(device)

    def hook(module, inputs, output):
        return replace_pseudo_embeddings(ids, output, pseudo, placeholder[0])

    handle = encoder.text_model.embeddings.token_embedding.register_forward_hook(hook)
    try:
        return encoder(input_ids=ids).last_hidden_state
    finally:
        handle.remove()


def trainable_kv(unet):
    if unet.config.in_channels != 4:
        raise ValueError("Trial requires a four-channel text-to-image UNet")
    unet.requires_grad_(False)
    selected = {}
    for name, parameter in unet.named_parameters():
        if ".attn2.to_k." in name or ".attn2.to_v." in name:
            parameter.requires_grad_(True)
            selected[name] = parameter
    if not selected:
        raise ValueError("No cross-attention K/V parameters found")
    return selected


def diffusion_loss(unet, scheduler, latents, hidden, generator):
    noise = torch.randn(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
    timesteps = torch.randint(scheduler.config.num_train_timesteps, (len(latents),),
                             device=latents.device, generator=generator)
    noisy = scheduler.add_noise(latents, noise, timesteps)
    prediction = unet(noisy, timesteps, encoder_hidden_states=hidden).sample
    kind = scheduler.config.prediction_type
    if kind == "epsilon":
        target = noise
    elif kind == "v_prediction":
        target = scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unsupported prediction target: {kind}")
    loss = (prediction.float() - target.float()).square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite diffusion loss")
    return loss


class TrialBudget:
    """Persist one deadline across cache, train, inference and interrupted resumes.

    Wall time from first GPU use is a conservative upper bound on one GPU's use.
    Caller must check around each bounded model operation, not only every epoch.
    """
    def __init__(self, path, seconds=28800):
        self.path = Path(path)
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state["maximum_seconds"] != seconds:
                raise ValueError("Cannot silently replace trial budget")
        else:
            self.state = dict(started_unix=time.time(), maximum_seconds=seconds)
            write_json(self.path, self.state)

    def remaining(self):
        return self.state["maximum_seconds"] - (time.time() - self.state["started_unix"])

    def check(self, reserve=0):
        if self.remaining() <= reserve:
            raise TimeoutError("Cumulative CIG trial time budget reached")
