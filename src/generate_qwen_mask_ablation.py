"""Generate matched Qwen-Image-Edit views for a mask ablation.

One raw Qwen result is shared by every requested output mode, so differences
between none, fixed, and CLIPSeg masks cannot be attributed to generation
randomness. The target image is never opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from tqdm import tqdm

from generate_local_edit_views import (
    image_path,
    stable_key,
    stratified_subset,
    valid_samples,
)
from generate_qwen_local_edit_views import (
    quantization_config,
    qwen_local_mask,
    qwen_prompt,
    qwen_region_kind,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fashioniq-root", default="../data/FashionIQ")
    parser.add_argument("--category", default="dress")
    parser.add_argument("--split", default="val")
    parser.add_argument("--structured-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen-Image-Edit")
    parser.add_argument("--cache-dir", default="../model_cache")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--mask-modes",
        default="none,fixed,clipseg",
        help="Comma-separated subset of none,fixed,clipseg.",
    )
    parser.add_argument(
        "--quantization", choices=("4bit", "8bit", "none"), default="4bit"
    )
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument(
        "--clipseg-model", default="CIDAS/clipseg-rd64-refined"
    )
    parser.add_argument("--clipseg-threshold", type=float, default=0.35)
    parser.add_argument(
        "--clipseg-dilation",
        type=float,
        default=0.04,
        help="Mask dilation radius as a fraction of output resolution.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def clipseg_prompts(category):
    return {
        "dress": ["dress", "clothing garment", "skirt"],
        "shirt": ["shirt", "upper body clothing", "sleeves"],
        "toptee": ["top", "t-shirt", "upper body clothing"],
    }.get(category, [category, "clothing garment"])


class DynamicGarmentMask:
    def __init__(self, model_name, cache_dir, threshold, dilation):
        from transformers import (
            CLIPSegForImageSegmentation,
            CLIPSegProcessor,
        )

        self.processor = CLIPSegProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            model_name, cache_dir=cache_dir
        ).eval()
        self.threshold = threshold
        self.dilation = dilation

    def __call__(self, image, prompts, fallback):
        inputs = self.processor(
            text=prompts,
            images=[image] * len(prompts),
            padding=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        probabilities = torch.sigmoid(logits).amax(dim=0, keepdim=True)
        probabilities = F.interpolate(
            probabilities[None],
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        binary = probabilities >= self.threshold
        coverage = float(binary.float().mean())
        if coverage < 0.015 or coverage > 0.85:
            return fallback, coverage, True

        mask = Image.fromarray(
            (binary.numpy().astype(np.uint8) * 255), mode="L"
        )
        radius = max(1, round(min(image.size) * self.dilation))
        mask = mask.filter(ImageFilter.MaxFilter(2 * radius + 1))
        mask = mask.filter(
            ImageFilter.GaussianBlur(radius=max(2, radius // 3))
        )
        return mask, coverage, False


def valid_image(image):
    pixels = np.asarray(image)
    return not (pixels.mean() < 5 or pixels.std() < 2)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    modes = [
        value.strip() for value in args.mask_modes.split(",") if value.strip()
    ]
    invalid = set(modes) - {"none", "fixed", "clipseg"}
    if invalid:
        raise ValueError(f"Unknown mask modes: {sorted(invalid)}")
    if not modes:
        raise ValueError("At least one mask mode is required")

    from diffusers import QwenImageEditPipeline

    candidates = valid_samples(
        args.structured_path, args.category, args.split
    )
    selected = stratified_subset(candidates, args.limit, args.seed)
    output_root = Path(args.output_dir)
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        (output_root / mode / "images").mkdir(parents=True, exist_ok=True)
        (output_root / mode / "masks").mkdir(parents=True, exist_ok=True)

    segmenter = None
    if "clipseg" in modes:
        segmenter = DynamicGarmentMask(
            args.clipseg_model,
            args.cache_dir,
            args.clipseg_threshold,
            args.clipseg_dilation,
        )

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "cache_dir": args.cache_dir,
        "low_cpu_mem_usage": True,
    }
    qconfig = quantization_config(args.quantization)
    if qconfig is not None:
        load_kwargs["quantization_config"] = qconfig
    pipe = QwenImageEditPipeline.from_pretrained(args.model, **load_kwargs)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=None)

    records = {mode: [] for mode in modes}
    prompts = clipseg_prompts(args.category)
    for index, sample in enumerate(tqdm(selected, desc="Qwen mask ablation")):
        candidate = sample["candidate"]
        target = sample["target"]
        filename = f"{candidate}__{target}"
        raw_path = raw_dir / f"{filename}.jpg"
        edit = sample["structured_edit"]
        kind = qwen_region_kind(edit)
        confidence = float(edit.get("confidence", {}).get("score", 0.0))
        with Image.open(image_path(args.fashioniq_root, candidate)) as handle:
            source = handle.convert("RGB").resize(
                (args.resolution, args.resolution)
            )
        fixed_mask = qwen_local_mask(source.size, kind)

        if args.resume and raw_path.exists():
            with Image.open(raw_path) as handle:
                raw = handle.convert("RGB")
            raw_valid = valid_image(raw)
        else:
            generator = torch.Generator(device="cuda").manual_seed(
                args.seed + index
            )
            with torch.inference_mode():
                raw = pipe(
                    image=source,
                    prompt=qwen_prompt(edit),
                    negative_prompt=" ",
                    true_cfg_scale=args.true_cfg_scale,
                    guidance_scale=1.0,
                    num_inference_steps=args.steps,
                    height=args.resolution,
                    width=args.resolution,
                    generator=generator,
                ).images[0].convert("RGB")
            raw_valid = valid_image(raw)
            if not raw_valid:
                raw = source
            raw.save(raw_path, quality=95)

        dynamic_mask = None
        dynamic_coverage = None
        dynamic_fallback = False
        if "clipseg" in modes:
            dynamic_mask, dynamic_coverage, dynamic_fallback = segmenter(
                source, prompts, fixed_mask
            )

        for mode in modes:
            if mode == "none":
                mask = Image.new("L", source.size, 255)
                result = raw
            elif mode == "fixed":
                mask = fixed_mask
                result = Image.composite(raw, source, mask)
            else:
                mask = dynamic_mask
                result = Image.composite(raw, source, mask)
            output_path = output_root / mode / "images" / f"{filename}.jpg"
            mask_path = output_root / mode / "masks" / f"{filename}.png"
            result.save(output_path, quality=95)
            mask.save(mask_path)
            records[mode].append(
                {
                    "candidate": candidate,
                    "target": target,
                    "generated_path": str(output_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                    "mask_mode": mode,
                    "region_kind": kind,
                    "confidence": confidence,
                    "seed": args.seed + index,
                    "generation_valid": raw_valid,
                    "clipseg_raw_coverage": dynamic_coverage,
                    "clipseg_fallback": dynamic_fallback,
                    "target_image_used": False,
                }
            )

        for mode in modes:
            manifest = {
                "description": (
                    "Leakage-free Qwen mask ablation sharing identical raw "
                    "generations across mask modes"
                ),
                "model": args.model,
                "split": args.split,
                "category": args.category,
                "mask_mode": mode,
                "parameters": vars(args),
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
