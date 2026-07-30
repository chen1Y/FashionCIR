"""Generate target-hypothesis images from structured text with Qwen-Image.

Neither the reference image nor the target image is opened. Candidate and
target ids are retained only to align generated hypotheses with evaluation
queries.
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
from generate_qwen_local_edit_views import quantization_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="dress")
    parser.add_argument("--split", default="val")
    parser.add_argument("--structured-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen-Image")
    parser.add_argument("--cache-dir", default="../model_cache")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--quantization", choices=("4bit", "8bit", "none"), default="4bit"
    )
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def generation_prompt(edit, category):
    target = str(edit.get("target_description", "")).strip()
    additions = "; ".join(
        str(item.get("attribute", "")).strip()
        for item in edit.get("add") or []
        if item.get("attribute")
    )
    garment = {
        "dress": "dress",
        "shirt": "shirt",
        "toptee": "top or t-shirt",
    }.get(category, "fashion garment")
    return (
        f"A realistic ecommerce catalog photograph of one {garment}. "
        f"The garment is: {target}. Important visible attributes: {additions}. "
        "Show the complete garment clearly on one adult fashion model, centered "
        "front view, natural standing pose, plain white studio background, "
        "soft even lighting, accurate fabric texture, realistic proportions, "
        "high detail. No collage, no duplicate person, no text, no watermark."
    )


def valid_image(image):
    pixels = np.asarray(image)
    return not (pixels.mean() < 5 or pixels.std() < 2)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    from diffusers import QwenImagePipeline

    candidates = valid_samples(
        args.structured_path, args.category, args.split
    )
    selected = stratified_subset(candidates, args.limit, args.seed)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "cache_dir": args.cache_dir,
        "low_cpu_mem_usage": True,
    }
    qconfig = quantization_config(args.quantization)
    if qconfig is not None:
        load_kwargs["quantization_config"] = qconfig
    pipe = QwenImagePipeline.from_pretrained(args.model, **load_kwargs)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=None)

    records = []
    for index, sample in enumerate(tqdm(selected, desc="Qwen text views")):
        candidate = sample["candidate"]
        target = sample["target"]
        edit = sample["structured_edit"]
        output_path = image_dir / f"{candidate}__{target}.jpg"
        prompt = generation_prompt(edit, args.category)
        if args.resume and output_path.exists():
            from PIL import Image

            with Image.open(output_path) as handle:
                result = handle.convert("RGB")
            generation_valid = valid_image(result)
        else:
            generator = torch.Generator(device="cuda").manual_seed(
                args.seed + index
            )
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=(
                        "blurry, low quality, distorted body, extra limbs, "
                        "multiple people, cropped garment, text, watermark"
                    ),
                    true_cfg_scale=args.true_cfg_scale,
                    guidance_scale=1.0,
                    num_inference_steps=args.steps,
                    height=args.resolution,
                    width=args.resolution,
                    generator=generator,
                ).images[0].convert("RGB")
            generation_valid = valid_image(result)
            result.save(output_path, quality=95)

        confidence = float(
            edit.get("confidence", {}).get("score", 0.0)
        )
        records.append(
            {
                "candidate": candidate,
                "target": target,
                "generated_path": str(output_path.resolve()),
                "prompt": prompt,
                "confidence": confidence,
                "seed": args.seed + index,
                "generation_valid": generation_valid,
                "source_image_used": False,
                "target_image_used": False,
            }
        )
        manifest = {
            "description": (
                "Leakage-free target hypotheses generated from structured "
                "text only; no FashionIQ image is opened by the generator"
            ),
            "model": args.model,
            "split": args.split,
            "category": args.category,
            "parameters": vars(args),
            "selection_digest": stable_key(selected[0], args.seed),
            "records": records,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    print(
        json.dumps(
            {
                "generated": len(records),
                "manifest": str(output_dir / "manifest.json"),
                "max_gpu_gib": (
                    torch.cuda.max_memory_allocated() / 1024**3
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
