"""Run one bounded, train-only FashionIQ SD1.5 CIG-inspired experiment.

Assets are downloaded separately. Training never downloads, changes a protocol,
reads official validation targets, or restarts a completed trial automatically.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cig_trial import (Phi, TrialBudget, diffusion_loss, encode_prompts, file_sha,
                       mask_keyword, trainable_kv, write_json)
from experiment_storage import atomic_torch_save, require_space
from fashioniq_augmentation import context
from fashioniq_noise import dataset_fingerprint, image_path


def prepare_manifest(args):
    """All category held-out IDs and image bytes are excluded before sampling."""
    from nltk.tag import PerceptronTagger
    tagger = PerceptronTagger(lang="eng")
    ctx = context(args.root, "dress")
    audit = json.loads(args.noise_audit.read_text())
    if audit["dataset_fingerprint"] != dataset_fingerprint(args.root, "dress"):
        raise ValueError("Stale FashionIQ noise audit")
    blocked = {key for key, value in audit["images"].items() if value["flags"] or value["hard_errors"]}
    heldout = set()
    for cat in ("dress", "shirt", "toptee"):
        for split in ("val", "test"):
            for key in json.loads((args.root / "image_splits" / f"split.{cat}.{split}.json").read_text()):
                path = image_path(args.root, cat, key)
                if path:
                    heldout.add(file_sha(path))
    captions = json.loads((args.root / "captions" / "image_captions_dress_train.json").read_text())
    ids = sorted(ctx["allowed"] - blocked)
    random.Random(42).shuffle(ids)
    rows, seen = [], set(heldout)
    for key in ids:
        caption = captions.get(key, "")
        if not isinstance(caption, str) or not caption.strip() or "$" in caption:
            continue
        path = image_path(args.root, "dress", key)
        if not path:
            continue
        digest = file_sha(path)
        if digest in seen:
            continue
        try:
            masked, removed = mask_keyword(caption, tagger)
        except ValueError:
            continue
        if not masked:
            continue
        seen.add(digest)
        rows.append(dict(image_id=key, image_path=str(path), sha256=digest, caption=caption,
                         masked_caption=masked, removed_keyword=removed,
                         split="train" if len(rows) < args.train_images else "internal_dev"))
        if len(rows) == args.train_images + args.dev_images:
            break
    if len(rows) != args.train_images + args.dev_images:
        raise ValueError("Insufficient clean train-only image-caption pairs")
    return dict(seed=42, source_signature=ctx["source_signature"], rows=rows,
                official_heldout_byte_hashes=len(heldout), official_heldout_overlap=0,
                train_count=args.train_images, internal_dev_count=args.dev_images,
                masking="NLTK first adjective* noun+ span; remove orphaned initial determiner",
                tagger_sha256=file_sha(args.assets / "averaged_perceptron_tagger_eng.zip"))


def wait_for_gpu(args):
    """Wait for the already-authorized FashionIQ task; never signal its processes."""
    while True:
        state_path = args.fashioniq_results / "followup_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get("status", "").startswith(("failed", "stopped")):
            raise RuntimeError("FashionIQ task stopped; inspect before GPU handoff")
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        if state.get("status") == "complete" and not processes:
            return
        write_json(args.output / "trial_state.json", dict(status="waiting_for_fashioniq", fashioniq=state))
        time.sleep(30)


def prepare_cache(args, manifest, budget):
    import torch
    from PIL import Image, ImageOps
    import numpy as np
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
    from safetensors.torch import save_file
    cache = args.output / "conditioning.safetensors"
    if cache.exists():
        raise FileExistsError("Existing cache requires explicit review, not blind regeneration")
    budget.check(600)
    tokenizer = CLIPTokenizer.from_pretrained(args.assets / "sd15" / "tokenizer", local_files_only=True)
    text = CLIPTextModel.from_pretrained(args.assets / "sd15" / "text_encoder", local_files_only=True).eval().cuda().requires_grad_(False)
    vision = CLIPVisionModelWithProjection.from_pretrained(args.assets / "clip_l14", local_files_only=True).eval().cuda().requires_grad_(False)
    processor = CLIPImageProcessor.from_pretrained(args.assets / "clip_l14", local_files_only=True)
    phi = Phi()
    payload = torch.load(args.assets / "SEARLE_ViT-L14.pt", map_location="cpu", weights_only=True)
    phi.load_state_dict(payload["Phi"], strict=True)
    phi.eval().cuda().requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(args.assets / "sd15" / "vae", local_files_only=True).eval().cuda().requires_grad_(False)
    latents, conditions, unmasked, pseudo_tokens = [], [], [], []
    with torch.inference_mode():
        for index, row in enumerate(manifest["rows"]):
            budget.check(120)
            if index % 32 == 0:
                require_space(args.output)
                write_json(args.output / "trial_state.json", dict(status="caching", done=index, total=len(manifest["rows"])))
                print("cache", index, flush=True)
            if file_sha(row["image_path"]) != row["sha256"]:
                raise ValueError("Image bytes changed after manifest lock")
            with Image.open(row["image_path"]) as source:
                image = source.convert("RGB")
                # Domain pilot uses the standard CLIP center crop (not SEARLE targetpad).
                image_features = vision(**processor(images=image, return_tensors="pt").to("cuda")).image_embeds
                pseudo = phi(image_features.float())
                hidden = encode_prompts(tokenizer, text, pseudo, [row["masked_caption"]])
                full = encode_prompts(tokenizer, text, pseudo, [row["caption"]])
                rgb = ImageOps.fit(image, (512, 512), method=Image.Resampling.LANCZOS)
                pixels = torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1
                # Fixed posterior mode saves a deterministic cache; noise remains stochastic in training.
                latent = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
            latents.append(latent.cpu().half())
            conditions.append(hidden.cpu().half())
            unmasked.append(full.cpu().half())
            pseudo_tokens.append(pseudo.cpu().float())
        tensors = dict(latents=torch.cat(latents), hidden=torch.cat(conditions),
                       unmasked_hidden=torch.cat(unmasked), pseudo=torch.cat(pseudo_tokens))
        save_file(tensors, str(cache), metadata={"manifest_sha256": file_sha(args.output / "manifest.json"),
                                               "protocol_sha256": file_sha(args.output / "protocol.json")})
    del text, vision, processor, phi, vae, tensors
    gc.collect()
    torch.cuda.empty_cache()
    write_json(args.output / "cache_verified.json", dict(sha256=file_sha(cache), rows=len(latents),
                                                        manifest_sha256=file_sha(args.output / "manifest.json")))


def train(args, manifest, budget):
    import torch
    from diffusers import UNet2DConditionModel, DDPMScheduler
    from safetensors import safe_open
    from safetensors.torch import load_file
    cache_path = args.output / "conditioning.safetensors"
    recorded = json.loads((args.output / "cache_verified.json").read_text())
    if file_sha(cache_path) != recorded["sha256"]:
        raise ValueError("Conditioning cache integrity changed")
    with safe_open(str(cache_path), framework="pt") as stream:
        metadata = stream.metadata()
    if (metadata["manifest_sha256"] != file_sha(args.output / "manifest.json")
            or metadata["protocol_sha256"] != file_sha(args.output / "protocol.json")):
        raise ValueError("Conditioning cache is bound to another manifest/protocol")
    cache = load_file(str(cache_path))
    latents, hidden = cache["latents"], cache["hidden"]
    if len(latents) != len(manifest["rows"]) or tuple(latents.shape[1:]) != (4, 64, 64) or tuple(hidden.shape[1:]) != (77, 768):
        raise ValueError("Cached SD1.5 tensor layout invalid")
    unet = UNet2DConditionModel.from_pretrained(args.assets / "sd15" / "unet", local_files_only=True).cuda()
    selected = trainable_kv(unet)
    unet.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(selected.values(), lr=1e-6, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = DDPMScheduler.from_pretrained(args.assets / "sd15" / "scheduler", local_files_only=True)
    generator = torch.Generator(device="cuda").manual_seed(42)
    sampler = torch.Generator().manual_seed(42)
    n = manifest["train_count"]
    original = {key: value.detach().cpu().clone() for key, value in selected.items()}
    write_json(args.output / "trainable_parameters.json", dict(names=list(selected),
               count=sum(p.numel() for p in selected.values()), selection="attn2.to_k / attn2.to_v only"))

    def evaluate():
        unet.eval()
        dev_gen = torch.Generator(device="cuda").manual_seed(7331)
        values = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start in range(n, len(latents), 2):
                budget.check(90)
                values.append(diffusion_loss(unet, scheduler, latents[start:start+2].cuda().float(),
                                             hidden[start:start+2].cuda().float(), dev_gen).item())
        unet.train()
        return sum(values) / len(values)

    def save(name, step, dev_loss):
        require_space(args.output)
        weights = {key: value.detach().cpu() for key, value in selected.items()}
        delta = sum((weights[key] - original[key]).square().sum().item() for key in weights) ** 0.5
        payload = dict(kv_state=weights, optimizer=optimizer.state_dict(), step=step, dev_loss=dev_loss,
                       parameter_delta_l2=delta, protocol_sha256=file_sha(args.output / "protocol.json"),
                       manifest_sha256=file_sha(args.output / "manifest.json"), generator_state=generator.get_state(),
                       sampler_state=sampler.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        atomic_torch_save(payload, args.output / name)
        return delta

    baseline = evaluate()
    write_json(args.output / "baseline_denoising.json", dict(internal_dev_mse=baseline, seed=7331))
    print("baseline_internal_dev_mse", baseline, flush=True)
    best, done, dev = float("inf"), 0, baseline
    started = time.monotonic()
    try:
        for step in range(1, args.steps + 1):
            budget.check(300)
            require_space(args.output)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(8):
                budget.check(180)
                ids = torch.randint(n, (2,), generator=sampler)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = diffusion_loss(unet, scheduler, latents[ids].cuda().float(), hidden[ids].cuda().float(), generator)
                (loss / 8).backward()
                losses.append(loss.item())
            norm = torch.nn.utils.clip_grad_norm_(selected.values(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            done = step
            row = dict(step=step, train_mse=sum(losses)/8, grad_norm=float(norm),
                       elapsed_train_seconds=time.monotonic()-started, remaining_seconds=budget.remaining())
            if step == 1 or step % 100 == 0 or step == args.steps:
                dev = evaluate()
                row["internal_dev_mse"] = dev
                row["parameter_delta_l2"] = save("last.pt", step, dev)
                if dev < best:
                    best = dev
                    save("best.pt", step, dev)
            with (args.output / "training.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            write_json(args.output / "trial_state.json", dict(status="training", **row))
            print(json.dumps(row), flush=True)
    except (TimeoutError, KeyboardInterrupt):
        # An incomplete accumulation is discarded; only completed optimizer steps are checkpointed.
        save("budget_stop.pt", done, dev)
        write_json(args.output / "trial_state.json", dict(status="stopped_budget", completed_steps=done,
                   baseline_internal_dev_mse=baseline, last_internal_dev_mse=dev))
        return False
    write_json(args.output / "training_summary.json", dict(status="complete", completed_steps=done,
               baseline_internal_dev_mse=baseline, best_internal_dev_mse=best,
               metric_scope="Self-reconstruction noise MSE on internal train-image dev partition; not retrieval accuracy"))
    del unet, optimizer, cache
    gc.collect()
    torch.cuda.empty_cache()
    return True


def generate_checks(args, manifest, budget):
    """Fixed internal-dev qualitative samples, before/after, no retrieval claim."""
    import torch
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    from safetensors.torch import load_file
    budget.check(600)
    pipe = StableDiffusionPipeline.from_pretrained(args.assets / "sd15", safety_checker=None,
              feature_extractor=None, requires_safety_checker=False, torch_dtype=torch.float16, local_files_only=True).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    cache = load_file(str(args.output / "conditioning.safetensors"))
    picks = range(manifest["train_count"], manifest["train_count"] + 8)
    output = args.output / "qualitative"
    output.mkdir(exist_ok=True)
    pipe.set_progress_bar_config(disable=True)
    records = []
    with torch.inference_mode():
        for phase in ("base", "trained"):
            if phase == "trained":
                payload = torch.load(args.output / "best.pt", map_location="cpu", weights_only=True)
                selected = trainable_kv(pipe.unet)
                if selected.keys() != payload["kv_state"].keys():
                    raise ValueError("Checkpoint K/V layout mismatch")
                with torch.no_grad():
                    for name, parameter in selected.items():
                        parameter.copy_(payload["kv_state"][name].to(parameter))
            for index in picks:
                budget.check(120)
                require_space(args.output)
                image = pipe(prompt_embeds=cache["unmasked_hidden"][index:index+1].cuda().half(),
                             num_inference_steps=30, guidance_scale=7.5, height=512, width=512,
                             generator=torch.Generator("cuda").manual_seed(42000+index)).images[0]
                path = output / f"{phase}_{index}.png"
                image.save(path)
                records.append(dict(phase=phase, image_id=manifest["rows"][index]["image_id"],
                               source=manifest["rows"][index]["image_path"], caption=manifest["rows"][index]["caption"],
                               generated=str(path), sha256=file_sha(path), seed=42000+index))
    write_json(output / "records.json", dict(scope="8 fixed internal-dev reconstruction probes, not CIR retrieval evaluation", records=records))

    # Separate query-only editing probes: first 8 original training queries whose
    # reference belongs to the internal dev partition. Do not copy/read targets.
    from PIL import Image
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    vision = CLIPVisionModelWithProjection.from_pretrained(args.assets / "clip_l14", local_files_only=True).eval().cuda().requires_grad_(False)
    processor = CLIPImageProcessor.from_pretrained(args.assets / "clip_l14", local_files_only=True)
    phi = Phi()
    phi.load_state_dict(torch.load(args.assets / "SEARLE_ViT-L14.pt", map_location="cpu", weights_only=True)["Phi"])
    phi.eval().cuda().requires_grad_(False)
    dev_ids = {row["image_id"] for row in manifest["rows"] if row["split"] == "internal_dev"}
    queries = json.loads((args.root / "captions" / "cap.dress.train.json").read_text())
    query_records = []
    for index, query in enumerate(queries):
        if query["candidate"] not in dev_ids:
            continue
        budget.check(120)
        path = image_path(args.root, "dress", query["candidate"])
        delta = " and ".join(query["captions"])
        with torch.inference_mode(), Image.open(path) as image:
            pseudo = phi(vision(**processor(images=image.convert("RGB"), return_tensors="pt").to("cuda")).image_embeds.float())
            hidden = encode_prompts(pipe.tokenizer, pipe.text_encoder, pseudo, [delta])
            generated = pipe(prompt_embeds=hidden, num_inference_steps=30, guidance_scale=7.5,
                         height=512, width=512, generator=torch.Generator("cuda").manual_seed(63000+index)).images[0]
        generated_path = output / f"query_only_{index}.png"
        generated.save(generated_path)
        query_records.append(dict(query_index=index, reference_id=query["candidate"], reference_path=str(path),
                              delta=delta, seed=63000+index, generated=str(generated_path), sha256=file_sha(generated_path)))
        if len(query_records) == 8:
            break
    write_json(output / "query_only_records.json", dict(scope="8 fixed train-query editing probes using internal-dev references; no target images read, not retrieval evaluation", records=query_records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise-audit", type=Path, required=True)
    parser.add_argument("--fashioniq-results", type=Path, required=True)
    parser.add_argument("--train-images", type=int, default=1024)
    parser.add_argument("--dev-images", type=int, default=128)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if min(args.train_images, args.dev_images, args.steps) < 1 or args.steps > 500:
        raise ValueError("Trial image/step limits invalid")
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (args.output / "trial.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import diffusers
    import transformers
    import nltk
    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(2)
    protocol = dict(name="CIG-inspired SD1.5 FashionIQ domain pilot", seed=42, steps=args.steps,
        train_images=args.train_images, internal_dev_images=args.dev_images, learning_rate=1e-6,
        microbatch=2, accumulation=8, effective_batch=16, maximum_gpu_wall_seconds=28800,
        optimizer="AdamW", betas=[0.9, 0.999], weight_decay=0.01, gradient_clip_norm=1.0,
        software=dict(python=sys.version, torch=torch.__version__, diffusers=diffusers.__version__,
                      transformers=transformers.__version__, nltk=nltk.__version__),
        resolution=512, trainable="cross-attention attn2.to_k and attn2.to_v", cfg_training_dropout=0,
        phi="SEARLE_ViT-L14", phi_sha256=file_sha(args.assets / "SEARLE_ViT-L14.pt"),
        differences_from_paper=["FashionIQ train-only domain pilot rather than 490k CC3M images",
            "512 center crop and CLIP standard center crop rather than SEARLE targetpad",
            "deterministic VAE posterior mode cache", "NLTK noun-phrase mask", "single GPU BF16 mixed precision"],
        assets=json.loads((args.assets / "assets_verified.json").read_text()),
        code_sha256={name:file_sha(ROOT / name) for name in ("src/cig_trial.py", "scripts/run_cig_domain_trial.py")},
        retrieval_training_run=False)
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Protocol changed; use a separate reviewed experiment")
    else:
        write_json(protocol_path, protocol)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = prepare_manifest(args)
        write_json(manifest_path, manifest)
    if args.prepare_only:
        print("prepared", manifest["train_count"], manifest["internal_dev_count"], flush=True)
        return
    if (args.output / "gpu_budget.json").exists():
        raise RuntimeError("Existing GPU trial: inspect checkpoints/budget before any explicit resume")
    wait_for_gpu(args)
    require_space(args.output)
    budget = TrialBudget(args.output / "gpu_budget.json")
    import signal
    def deadline(signum, frame):
        raise TimeoutError("Hard eight-hour GPU wall deadline reached")
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, max(1, budget.remaining()))
    try:
        prepare_cache(args, manifest, budget)
        if not train(args, manifest, budget):
            return
        generate_checks(args, manifest, budget)
        write_json(args.output / "trial_state.json", dict(status="complete", completed_steps=args.steps,
                   gpu_wall_seconds=28800-budget.remaining(), retrieval_evaluated=False))
    except BaseException as error:
        write_json(args.output / "trial_state.json", dict(status="failed", error_type=type(error).__name__,
                   error=str(error), gpu_wall_seconds=28800-budget.remaining()))
        raise


if __name__ == "__main__":
    main()
