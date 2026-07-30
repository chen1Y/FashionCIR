"""Generate leakage-free, locally inpainted FashionIQ query views.

Only the reference image and its modification-derived structured JSON are used.
The target image is never opened.  The target id is retained solely as a lookup
key for later retrieval evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fashioniq-root", default="../data/FashionIQ")
    parser.add_argument("--category", default="dress")
    parser.add_argument("--split", default="val")
    parser.add_argument("--structured-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model",
        default="stable-diffusion-v1-5/stable-diffusion-inpainting",
    )
    parser.add_argument("--cache-dir", default="../model_cache")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.78)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def stable_key(sample, seed):
    text = f"{seed}:{sample['candidate']}:{sample['target']}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def valid_samples(path, category, split):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = payload.get("samples", payload)
    output = []
    for sample in samples:
        edit = sample.get("structured_edit", {})
        if (
            sample.get("category") == category
            and sample.get("split") == split
            and sample.get("status") == "ok"
            and sample.get("validation", {}).get("valid", False)
            and sample.get("semantic_validation", {}).get("valid", False)
            and edit.get("add")
        ):
            output.append(sample)
    return output


def stratified_subset(samples, limit, seed):
    """Round-robin over the primary requested edit category."""
    groups = {}
    for sample in samples:
        additions = sample["structured_edit"].get("add") or []
        category = str(additions[0].get("category", "other")).lower()
        groups.setdefault(category, []).append(sample)
    for values in groups.values():
        values.sort(key=lambda item: stable_key(item, seed))
    selected = []
    keys = sorted(groups)
    while keys and (limit <= 0 or len(selected) < limit):
        next_keys = []
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                if limit > 0 and len(selected) >= limit:
                    break
            if groups[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def image_path(root, image_id):
    image_id = image_id.removeprefix("dress_").removeprefix("shirt_")
    image_id = image_id.removeprefix("toptee_")
    # Match newDataset.FashionIQ exactly; do not silently use another resize.
    return Path(root) / "resized_image" / f"{image_id}.jpg"


def region_kind(edit):
    text = " ".join(
        str(item.get("region", ""))
        for field in ("add", "remove")
        for item in edit.get(field) or []
    ).lower()
    if any(word in text for word in ("neck", "shoulder", "sleeve", "arm", "chest")):
        return "upper"
    if any(word in text for word in ("hem", "skirt", "lower", "leg")):
        return "lower"
    if any(word in text for word in ("waist", "torso", "hip", "bodice")):
        return "middle"
    return "full"


def local_mask(size, kind):
    """Conservative soft garment-shaped mask in normalized product-image space."""
    width, height = size
    boxes = {
        # Keep the face/hair outside the editable area.  FashionIQ dress
        # photos often include a person, and identity changes are pure noise
        # for garment retrieval.
        "upper": (0.13, 0.20, 0.87, 0.57),
        "middle": (0.18, 0.22, 0.82, 0.76),
        "lower": (0.15, 0.42, 0.85, 0.97),
        "full": (0.15, 0.18, 0.85, 0.97),
    }
    left, top, right, bottom = boxes[kind]
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        (
            int(left * width),
            int(top * height),
            int(right * width),
            int(bottom * height),
        ),
        fill=255,
    )
    return mask.filter(ImageFilter.GaussianBlur(radius=max(4, width // 64)))


def edit_prompt(edit):
    target = str(edit.get("target_description", "")).strip()
    additions = ", ".join(
        str(item.get("attribute", "")).strip()
        for item in edit.get("add") or []
        if item.get("attribute")
    )
    removals = ", ".join(
        str(item.get("attribute", "")).strip()
        for item in edit.get("remove") or []
        if item.get("attribute")
    )
    prompt = (
        f"Professional ecommerce catalog photograph of the same garment. "
        f"Edit it to become: {target}. Requested additions: {additions}. "
    )
    if removals:
        prompt += f"Remove or replace: {removals}. "
    return prompt + (
        "Preserve the original pose, framing, background, lighting, and every "
        "unmentioned garment detail. One centered dress, realistic fabric."
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    from diffusers import StableDiffusionInpaintPipeline

    candidates = valid_samples(args.structured_path, args.category, args.split)
    selected = stratified_subset(candidates, args.limit, args.seed)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=args.cache_dir,
    ).to("cuda")
    pipe.enable_attention_slicing()
    negative = (
        "person, mannequin, multiple garments, text, watermark, logo, distorted "
        "garment, extra sleeves, duplicated details, changed background"
    )
    records = []
    for index, sample in enumerate(tqdm(selected, desc="local edits")):
        candidate = sample["candidate"]
        target = sample["target"]
        source_path = image_path(args.fashioniq_root, candidate)
        output_path = image_dir / f"{candidate}__{target}.jpg"
        mask_path = mask_dir / f"{candidate}__{target}.png"
        edit = sample["structured_edit"]
        kind = region_kind(edit)
        confidence = float(edit.get("confidence", {}).get("score", 0.0))
        if not (args.resume and output_path.exists()):
            with Image.open(source_path) as handle:
                source = handle.convert("RGB").resize((512, 512))
            mask = local_mask(source.size, kind)
            generator = torch.Generator(device="cuda").manual_seed(
                args.seed + index
            )
            result = pipe(
                prompt=edit_prompt(edit),
                negative_prompt=negative,
                image=source,
                mask_image=mask,
                strength=args.strength,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]
            pixels = np.asarray(result)
            generation_valid = not (
                pixels.mean() < 5 or pixels.std() < 2
            )
            if not generation_valid:
                # A safety-filtered black frame is not an edited view.
                # Preserve a zero-residual fallback and record the event.
                result = source
            result.save(output_path, quality=95)
            mask.save(mask_path)
        else:
            with Image.open(output_path) as handle:
                pixels = np.asarray(handle.convert("RGB"))
            generation_valid = not (
                pixels.mean() < 5 or pixels.std() < 2
            )
        records.append(
            {
                "candidate": candidate,
                "target": target,
                "generated_path": str(output_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "region_kind": kind,
                "confidence": confidence,
                "seed": args.seed + index,
                "target_image_used": False,
                "generation_valid": generation_valid,
            }
        )
        manifest = {
            "description": "Leakage-free local edits from reference image + structured text",
            "model": args.model,
            "split": args.split,
            "category": args.category,
            "parameters": vars(args),
            "records": records,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    print(json.dumps({"generated": len(records), "manifest": str(output_dir / "manifest.json")}))


if __name__ == "__main__":
    main()
