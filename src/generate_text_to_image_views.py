"""Generate leakage-free text-to-image views for FashionIQ retrieval.

Two prompt routes share the same sample order and random seed:
1. the original relative FashionIQ modification text;
2. Qwen's structured target description, which also contains retained details.
The target image is never opened or passed to the generator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from generate_local_edit_views import (
    stable_key,
    stratified_subset,
    valid_samples,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="dress")
    parser.add_argument("--split", default="val")
    parser.add_argument("--structured-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model", default="stabilityai/stable-diffusion-xl-base-1.0"
    )
    parser.add_argument("--cache-dir", default="../model_cache")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument(
        "--prompt-modes",
        default="modification,structured",
        help="Comma-separated subset of modification,structured.",
    )
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def prompt_for(sample, mode):
    edit = sample["structured_edit"]
    if mode == "modification":
        description = sample["modification_text"].strip().rstrip(".")
        garment = (
            f"a women's {sample.get('category', 'fashion garment')} that "
            f"{description}"
        )
    else:
        description = str(edit.get("target_description", "")).strip()
        if not description:
            additions = ", ".join(
                item["attribute"]
                for item in edit.get("add") or []
                if item.get("attribute")
            )
            retained = ", ".join(
                item["attribute"]
                for item in edit.get("retain") or []
                if item.get("attribute")
            )
            description = ", ".join(
                value for value in (additions, retained) if value
            )
        garment = description
    return (
        f"Professional ecommerce catalog photograph of {garment}. "
        "One garment worn by one fashion model, full garment visible, centered, "
        "front view, natural proportions, clean plain light background, soft "
        "studio lighting, realistic fabric texture, high detail, no accessories."
    )


def valid_image(image):
    pixels = np.asarray(image.convert("RGB"))
    return not (pixels.mean() < 5 or pixels.std() < 2)


def restore_text_to_image_input(unet):
    """Restore the four latent input channels of an SD1.5 inpainting UNet."""
    if unet.config.in_channels != 9:
        return False
    old = unet.conv_in
    new = torch.nn.Conv2d(
        4,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    ).to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.copy_(old.weight[:, :4])
        if old.bias is not None:
            new.bias.copy_(old.bias)
    unet.conv_in = new
    unet.register_to_config(in_channels=4)
    return True


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    modes = [
        value.strip()
        for value in args.prompt_modes.split(",")
        if value.strip()
    ]
    invalid = set(modes) - {"modification", "structured"}
    if invalid:
        raise ValueError(f"Unknown prompt modes: {sorted(invalid)}")
    if not modes:
        raise ValueError("At least one prompt mode is required")

    from diffusers import AutoPipelineForText2Image

    candidates = valid_samples(
        args.structured_path, args.category, args.split
    )
    selected = stratified_subset(candidates, args.limit, args.seed)
    output_root = Path(args.output_dir)
    for mode in modes:
        (output_root / mode / "images").mkdir(parents=True, exist_ok=True)

    load_kwargs = {
        "torch_dtype": torch.float16,
        "cache_dir": args.cache_dir,
        "safety_checker": None,
        "requires_safety_checker": False,
        "variant": "fp16",
    }
    if args.model == "stabilityai/stable-diffusion-xl-base-1.0":
        load_kwargs["use_safetensors"] = True
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model, **load_kwargs
    )
    restored_inpaint_unet = restore_text_to_image_input(pipe.unet)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=None)

    negative_prompt = (
        "text, words, watermark, logo, collage, multiple people, multiple "
        "garments, duplicate body, cropped garment, distorted anatomy, blurry, "
        "painting, illustration, low detail"
    )
    records = {mode: [] for mode in modes}
    for index, sample in enumerate(tqdm(selected, desc="Text-to-image views")):
        candidate = sample["candidate"]
        target = sample["target"]
        filename = f"{candidate}__{target}.jpg"
        pending = [
            mode
            for mode in modes
            if not (
                args.resume
                and (output_root / mode / "images" / filename).exists()
            )
        ]
        if pending:
            prompts = [prompt_for(sample, mode) for mode in pending]
            generators = [
                torch.Generator(device="cuda").manual_seed(
                    args.seed + index
                )
                for _ in pending
            ]
            with torch.inference_mode():
                images = pipe(
                    prompt=prompts,
                    negative_prompt=[negative_prompt] * len(pending),
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    height=args.resolution,
                    width=args.resolution,
                    generator=generators,
                ).images
            for mode, image in zip(pending, images):
                image.convert("RGB").save(
                    output_root / mode / "images" / filename, quality=95
                )

        confidence = float(
            sample["structured_edit"].get("confidence", {}).get("score", 0.0)
        )
        for mode in modes:
            output_path = output_root / mode / "images" / filename
            from PIL import Image

            with Image.open(output_path) as handle:
                generation_valid = valid_image(handle)
            records[mode].append(
                {
                    "candidate": candidate,
                    "target": target,
                    "generated_path": str(output_path.resolve()),
                    "prompt_mode": mode,
                    "generation_prompt": prompt_for(sample, mode),
                    "confidence": confidence,
                    "seed": args.seed + index,
                    "generation_valid": generation_valid,
                    "target_image_used": False,
                }
            )

        for mode in modes:
            manifest = {
                "description": (
                    "Leakage-free FashionIQ text-to-image view generated from "
                    f"{mode} text only"
                ),
                "model": args.model,
                "split": args.split,
                "category": args.category,
                "prompt_mode": mode,
                "parameters": vars(args),
                "restored_inpaint_unet": restored_inpaint_unet,
                "selection_digest": stable_key(selected[0], args.seed),
                "records": records[mode],
            }
            (output_root / mode / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )

    print(
        json.dumps(
            {
                "generated": len(selected),
                "modes": modes,
                "output_root": str(output_root),
                "max_gpu_gib": (
                    torch.cuda.max_memory_allocated() / 1024**3
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
