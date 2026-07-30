"""Generate leakage-free FashionIQ local edits with Qwen-Image-Edit.

This reuses the exact sample selection, prompts, and face-protected masks from
the SD1.5 pilot.  The target id is only an evaluation key; the target image is
never opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

from generate_local_edit_views import (
    image_path,
    stable_key,
    stratified_subset,
    valid_samples,
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
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--edit-mode",
        choices=("global-composite", "inpaint"),
        default="global-composite",
        help=(
            "global-composite lets Qwen perform semantic editing, then restores "
            "all pixels outside the structured region mask from the source."
        ),
    )
    parser.add_argument(
        "--quantization", choices=("4bit", "8bit", "none"), default="4bit"
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Keep inactive pipeline components on CPU to reduce GPU memory.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def quantization_config(mode):
    if mode == "none":
        return None
    from diffusers.quantizers import PipelineQuantizationConfig

    kwargs = {}
    if mode == "4bit":
        kwargs = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": True,
        }
    else:
        kwargs = {"load_in_8bit": True}
    return PipelineQuantizationConfig(
        quant_backend=f"bitsandbytes_{mode}",
        quant_kwargs=kwargs,
        components_to_quantize=["transformer", "text_encoder"],
    )


def qwen_prompt(edit):
    # Keep the positive target state early and avoid ambiguous instructions.
    target = str(edit.get("target_description", "")).strip()
    additions = "; ".join(
        f"{item.get('attribute', '')} at {item.get('region', 'garment')}"
        for item in edit.get("add") or []
        if item.get("attribute")
    )
    removals = "; ".join(
        f"{item.get('attribute', '')} from {item.get('region', 'garment')}"
        for item in edit.get("remove") or []
        if item.get("attribute")
    )
    prompt = (
        "Edit only the masked garment region in this ecommerce image. "
        f"The final garment should be: {target}. Add or change: {additions}. "
    )
    if removals:
        prompt += f"Remove or replace: {removals}. "
    return prompt + (
        "Keep the same person, face, body, pose, garment identity, background, "
        "lighting, camera and all unmentioned details exactly unchanged. "
        "Produce a realistic catalog photograph without text or watermark."
    )


def qwen_region_kind(edit):
    regions = [
        str(item.get("region", "")).lower()
        for field in ("add", "remove")
        for item in edit.get(field) or []
    ]
    text = " ".join(regions)
    if any(word in text for word in ("full dress", "entire", "full garment")):
        return "full"
    if any(word in text for word in ("neck", "shoulder", "sleeve", "arm", "chest")):
        return "upper"
    if any(word in text for word in ("hem", "skirt", "lower", "leg")):
        return "lower"
    if any(word in text for word in ("waist", "torso", "hip", "bodice")):
        return "middle"
    return "full"


def qwen_local_mask(size, kind):
    """Region box that includes garment boundaries needed for shape changes."""
    width, height = size
    boxes = {
        "upper": (0.03, 0.18, 0.97, 0.62),
        "middle": (0.08, 0.20, 0.92, 0.82),
        "lower": (0.02, 0.38, 0.98, 0.99),
        "full": (0.02, 0.16, 0.98, 0.99),
    }
    left, top, right, bottom = boxes[kind]
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(
        (
            int(left * width),
            int(top * height),
            int(right * width),
            int(bottom * height),
        ),
        radius=max(8, width // 32),
        fill=255,
    )
    return mask.filter(ImageFilter.GaussianBlur(radius=max(2, width // 128)))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch_size < 1:
        raise ValueError("batch-size must be at least 1")
    from diffusers import QwenImageEditInpaintPipeline, QwenImageEditPipeline

    candidates = valid_samples(
        args.structured_path, args.category, args.split
    )
    selected = stratified_subset(candidates, args.limit, args.seed)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "cache_dir": args.cache_dir,
        "low_cpu_mem_usage": True,
    }
    qconfig = quantization_config(args.quantization)
    if qconfig is not None:
        load_kwargs["quantization_config"] = qconfig
    pipeline_class = (
        QwenImageEditInpaintPipeline
        if args.edit_mode == "inpaint"
        else QwenImageEditPipeline
    )
    pipe = pipeline_class.from_pretrained(args.model, **load_kwargs)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=None)

    records = []
    progress = tqdm(total=len(selected), desc="Qwen local edits")
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start : start + args.batch_size]
        batch_records = {}
        work = []
        for offset, sample in enumerate(batch):
            index = start + offset
            candidate = sample["candidate"]
            target = sample["target"]
            source_path = image_path(args.fashioniq_root, candidate)
            output_path = image_dir / f"{candidate}__{target}.jpg"
            mask_path = mask_dir / f"{candidate}__{target}.png"
            edit = sample["structured_edit"]
            kind = qwen_region_kind(edit)
            confidence = float(
                edit.get("confidence", {}).get("score", 0.0)
            )
            common = {
                "candidate": candidate,
                "target": target,
                "generated_path": str(output_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "region_kind": kind,
                "confidence": confidence,
                "seed": args.seed + index,
                "target_image_used": False,
            }
            if args.resume and output_path.exists():
                with Image.open(output_path) as handle:
                    pixels = np.asarray(handle.convert("RGB"))
                common["generation_valid"] = not (
                    pixels.mean() < 5 or pixels.std() < 2
                )
                batch_records[index] = common
                continue
            with Image.open(source_path) as handle:
                source = handle.convert("RGB").resize(
                    (args.resolution, args.resolution)
                )
            mask = qwen_local_mask(source.size, kind)
            work.append(
                {
                    "index": index,
                    "source": source,
                    "mask": mask,
                    "prompt": qwen_prompt(edit),
                    "output_path": output_path,
                    "mask_path": mask_path,
                    "record": common,
                }
            )

        if work:
            generators = [
                torch.Generator(device="cuda").manual_seed(
                    args.seed + item["index"]
                )
                for item in work
            ]
            inputs = {
                "image": [item["source"] for item in work],
                "prompt": [item["prompt"] for item in work],
                "negative_prompt": [" "] * len(work),
                "true_cfg_scale": args.true_cfg_scale,
                "guidance_scale": 1.0,
                "num_inference_steps": args.steps,
                "height": args.resolution,
                "width": args.resolution,
                "generator": generators,
            }
            if args.edit_mode == "inpaint":
                inputs["mask_image"] = [item["mask"] for item in work]
                inputs["strength"] = args.strength
            with torch.inference_mode():
                outputs = pipe(**inputs).images
            for item, result in zip(work, outputs):
                result = result.convert("RGB")
                if args.edit_mode == "global-composite":
                    result = Image.composite(
                        result.resize(item["source"].size),
                        item["source"],
                        item["mask"],
                    )
                pixels = np.asarray(result)
                generation_valid = not (
                    pixels.mean() < 5 or pixels.std() < 2
                )
                if not generation_valid:
                    result = item["source"]
                result.save(item["output_path"], quality=95)
                item["mask"].save(item["mask_path"])
                item["record"]["generation_valid"] = generation_valid
                batch_records[item["index"]] = item["record"]

        records.extend(
            batch_records[index]
            for index in range(start, start + len(batch))
        )
        progress.update(len(batch))
        manifest = {
            "description": (
                "Leakage-free Qwen local edits from reference image + "
                "structured text"
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
    progress.close()
    print(
        json.dumps(
            {
                "generated": len(records),
                "manifest": str(output_dir / "manifest.json"),
                "max_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
            }
        )
    )


if __name__ == "__main__":
    main()
