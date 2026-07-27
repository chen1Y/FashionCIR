"""Evaluate FashionIQ structured edits with a frozen OpenCLIP model.

This is a diagnostic A/B test, not a trained-model comparison.  Every text
representation uses the same image encoder, text encoder, gallery, and fixed
reference-image fusion weight.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fashioniq-root", type=Path, default=Path("../data/FashionIQ"))
    parser.add_argument("--structured-json", type=Path, required=True)
    parser.add_argument("--category", default="dress")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--gallery",
        choices=("query-union", "split"),
        default="query-union",
        help="Use images occurring in the selected queries or the full split.",
    )
    parser.add_argument("--model", default="ViT-H-14")
    parser.add_argument("--pretrained", default="laion2B-s32B-b79K")
    parser.add_argument("--cache-dir", type=Path, default=Path("../model_cache"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-weight", type=float, default=0.5)
    parser.add_argument(
        "--image-weights",
        help="Optional comma-separated fusion-weight sweep, for example 0.25,0.5,0.75.",
    )
    parser.add_argument(
        "--target-weights",
        default="0.25,0.5,0.75",
        help="Target-description weights for raw/target text-feature hybrids.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show representation examples without loading CLIP.",
    )
    return parser.parse_args()


def join_attributes(items: Iterable[dict]) -> str:
    values = []
    for item in items:
        attribute = str(item.get("attribute", "")).strip()
        region = str(item.get("region", "")).strip()
        if attribute:
            values.append(f"{attribute} on {region}" if region else attribute)
    return "; ".join(values)


def join_relations(items: Iterable[dict]) -> str:
    values = []
    for item in items:
        subject = str(item.get("subject", "")).strip()
        relation = str(item.get("relation", "")).strip().replace("_", " ")
        obj = str(item.get("object", "")).strip()
        phrase = " ".join(value for value in (subject, relation, obj) if value)
        if phrase:
            values.append(phrase)
    return "; ".join(values)


def structured_template(edit: dict) -> str:
    parts = []
    for label, key in (("Keep", "retain"), ("Remove", "remove"), ("Add", "add")):
        value = join_attributes(edit.get(key, []))
        if value:
            parts.append(f"{label}: {value}.")
    relations = join_relations(edit.get("relations", []))
    if relations:
        parts.append(f"Relations: {relations}.")
    return " ".join(parts) or str(edit.get("target_description", "")).strip()


def field_prompts(edit: dict) -> dict[str, str]:
    prompts = {}
    retain = join_attributes(edit.get("retain", []))
    remove = join_attributes(edit.get("remove", []))
    add = join_attributes(edit.get("add", []))
    relations = join_relations(edit.get("relations", []))
    if retain:
        prompts["retain"] = f"The target fashion item keeps {retain}."
    if remove:
        prompts["remove"] = f"The target fashion item does not have {remove}."
    if add:
        prompts["add"] = f"The target fashion item has {add}."
    if relations:
        prompts["relations"] = f"In the target fashion item, {relations}."
    return prompts


def load_samples(path: Path, limit: int) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    samples = payload["samples"] if isinstance(payload, dict) else payload
    selected = [
        sample
        for sample in samples
        if sample.get("status") == "ok"
        and sample.get("validation", {}).get("valid", False)
        and sample.get("semantic_validation", {}).get("valid", False)
    ]
    return selected[:limit] if limit > 0 else selected


def image_path(root: Path, image_id: str) -> Path:
    return root / "resized_image" / f"{image_id}.jpg"


def load_gallery(root: Path, category: str, split: str, mode: str, samples: list[dict]) -> list[str]:
    if mode == "query-union":
        return sorted({sample[key] for sample in samples for key in ("candidate", "target")})
    split_path = root / "image_splits" / f"split.{category}.{split}.json"
    with split_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def encode_images(model, preprocess, paths: list[Path], batch_size: int, device: torch.device) -> torch.Tensor:
    batches = []
    for start in tqdm(range(0, len(paths), batch_size), desc="images"):
        images = []
        for path in paths[start : start + batch_size]:
            with Image.open(path) as image:
                images.append(preprocess(image.convert("RGB")))
        tensor = torch.stack(images).to(device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            batches.append(F.normalize(model.encode_image(tensor), dim=-1).float().cpu())
    return torch.cat(batches)


def encode_texts(model, tokenizer, texts: list[str], batch_size: int, device: torch.device) -> torch.Tensor:
    batches = []
    for start in tqdm(range(0, len(texts), batch_size), desc="texts", leave=False):
        tokens = tokenizer(texts[start : start + batch_size]).to(device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            batches.append(F.normalize(model.encode_text(tokens), dim=-1).float().cpu())
    return torch.cat(batches)


def compose(reference: torch.Tensor, text: torch.Tensor, image_weight: float) -> torch.Tensor:
    return F.normalize(image_weight * reference + (1.0 - image_weight) * text, dim=-1)


def ranks_for_queries(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    samples: list[dict],
    gallery_index: dict[str, int],
) -> np.ndarray:
    similarities = query_features @ gallery_features.T
    ranks = []
    for row, sample in enumerate(samples):
        similarities[row, gallery_index[sample["candidate"]]] = -torch.inf
        order = torch.argsort(similarities[row], descending=True)
        position = torch.nonzero(order == gallery_index[sample["target"]], as_tuple=False)
        ranks.append(int(position[0, 0]) + 1)
    return np.asarray(ranks, dtype=np.int64)


def recalls(ranks: np.ndarray) -> dict[str, float]:
    return {f"R@{k}": float(np.mean(ranks <= k) * 100.0) for k in (1, 10, 50)}


def bootstrap_difference(
    ranks: np.ndarray,
    baseline_ranks: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    count = len(ranks)
    output = {}
    for k in (1, 10, 50):
        paired = (ranks <= k).astype(np.float32) - (baseline_ranks <= k).astype(np.float32)
        observed = float(paired.mean() * 100.0)
        if samples > 0:
            indices = rng.integers(0, count, size=(samples, count))
            draws = paired[indices].mean(axis=1) * 100.0
            low, high = np.percentile(draws, [2.5, 97.5])
        else:
            low = high = observed
        output[f"R@{k}"] = {
            "difference_points": observed,
            "ci95_low": float(low),
            "ci95_high": float(high),
        }
    return output


def main() -> None:
    args = parse_args()
    image_weights = (
        [float(value) for value in args.image_weights.split(",")]
        if args.image_weights
        else [args.image_weight]
    )
    target_weights = [float(value) for value in args.target_weights.split(",")]
    if not image_weights or any(not 0.0 <= value <= 1.0 for value in image_weights):
        raise ValueError("image weights must be in [0, 1]")
    if not target_weights or any(not 0.0 <= value <= 1.0 for value in target_weights):
        raise ValueError("target weights must be in [0, 1]")
    root = args.fashioniq_root.resolve()
    structured_path = args.structured_json.resolve()
    samples = load_samples(structured_path, args.limit)
    if not samples:
        raise ValueError("No valid structured samples were found")

    representations = {
        "raw_prompt": [
            sample["modification_text"]
            for sample in samples
        ],
        "target_description": [
            sample["structured_edit"]["target_description"] for sample in samples
        ],
        "structured_template": [
            structured_template(sample["structured_edit"]) for sample in samples
        ],
    }
    fields = [field_prompts(sample["structured_edit"]) for sample in samples]
    gallery_ids = load_gallery(root, args.category, args.split, args.gallery, samples)

    missing = [
        str(image_path(root, image_id))
        for image_id in set(gallery_ids)
        | {sample["candidate"] for sample in samples}
        if not image_path(root, image_id).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first: {missing[0]}")

    print(f"queries={len(samples)} gallery={len(gallery_ids)} mode={args.gallery}")
    for name, texts in representations.items():
        print(f"{name}_example={texts[0]}")
    print(f"field_prompts_example={fields[0]}")
    if args.dry_run:
        return

    import open_clip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        cache_dir=str(args.cache_dir.resolve()),
    )
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(args.model)

    gallery_index = {image_id: index for index, image_id in enumerate(gallery_ids)}
    gallery_features = encode_images(
        model,
        preprocess,
        [image_path(root, image_id) for image_id in gallery_ids],
        args.batch_size,
        device,
    )
    reference_features = gallery_features[
        torch.tensor([gallery_index[sample["candidate"]] for sample in samples])
    ]

    text_features = {
        name: encode_texts(model, tokenizer, texts, args.batch_size, device)
        for name, texts in representations.items()
    }
    for target_weight in target_weights:
        text_features[f"raw_target_hybrid_t{target_weight:g}"] = F.normalize(
            (1.0 - target_weight) * text_features["raw_prompt"]
            + target_weight * text_features["target_description"],
            dim=-1,
        )

    field_weights = {"retain": 0.15, "remove": 0.25, "add": 0.45, "relations": 0.15}
    flat_texts = []
    flat_keys = []
    for sample_index, sample_fields in enumerate(fields):
        for field_name, prompt in sample_fields.items():
            flat_keys.append((sample_index, field_name))
            flat_texts.append(prompt)
    flat_features = encode_texts(model, tokenizer, flat_texts, args.batch_size, device)
    field_feature_rows = []
    offset = 0
    for sample_index, sample_fields in enumerate(fields):
        weighted = []
        weights = []
        for field_name in sample_fields:
            assert flat_keys[offset] == (sample_index, field_name)
            weighted.append(flat_features[offset] * field_weights[field_name])
            weights.append(field_weights[field_name])
            offset += 1
        field_feature_rows.append(F.normalize(sum(weighted) / sum(weights), dim=-1))
    field_features = torch.stack(field_feature_rows)

    strategies = {}
    for name, features in text_features.items():
        strategies[f"{name}_text_only"] = features
        for image_weight in image_weights:
            strategies[f"{name}_image_fusion_w{image_weight:g}"] = compose(
                reference_features, features, image_weight
            )
    for image_weight in image_weights:
        raw_image_query = strategies[f"raw_prompt_image_fusion_w{image_weight:g}"]
        for target_weight in target_weights:
            strategies[
                f"raw_image_target_residual_t{target_weight:g}_image_fusion_w{image_weight:g}"
            ] = F.normalize(
                (1.0 - target_weight) * raw_image_query
                + target_weight * text_features["target_description"],
                dim=-1,
            )
    strategies["field_fusion_text_only"] = field_features
    for image_weight in image_weights:
        strategies[f"field_fusion_image_fusion_w{image_weight:g}"] = compose(
            reference_features, field_features, image_weight
        )

    ranks = {
        name: ranks_for_queries(features, gallery_features, samples, gallery_index)
        for name, features in strategies.items()
    }
    results = {
        "configuration": {
            "structured_json": str(structured_path),
            "category": args.category,
            "split": args.split,
            "gallery": args.gallery,
            "query_count": len(samples),
            "gallery_count": len(gallery_ids),
            "model": args.model,
            "pretrained": args.pretrained,
            "image_weights": image_weights,
            "target_weights": target_weights,
            "comparison": (
                "Text-only methods use raw_prompt_text_only as baseline; "
                "each image-fusion method uses raw_prompt at the same image weight."
            ),
            "seed": args.seed,
        },
        "metrics": {},
        "queries": [],
    }
    for name, strategy_ranks in ranks.items():
        if name.endswith("_text_only"):
            baseline_name = "raw_prompt_text_only"
        else:
            weight_suffix = name.rsplit("_w", 1)[1]
            baseline_name = f"raw_prompt_image_fusion_w{weight_suffix}"
        results["metrics"][name] = {
            **recalls(strategy_ranks),
            "median_rank": float(np.median(strategy_ranks)),
            "mean_rank": float(np.mean(strategy_ranks)),
            "comparison_baseline": baseline_name,
            "difference_vs_mode_matched_baseline": bootstrap_difference(
                strategy_ranks,
                ranks[baseline_name],
                args.bootstrap_samples,
                args.seed,
            ),
        }
        print(name, json.dumps(results["metrics"][name], ensure_ascii=False))

    for index, sample in enumerate(samples):
        results["queries"].append(
            {
                "query_id": sample["query_id"],
                "source_index": sample["source_index"],
                "candidate": sample["candidate"],
                "target": sample["target"],
                "ranks": {name: int(values[index]) for name, values in ranks.items()},
            }
        )

    output = args.output or structured_path.with_name(
        f"structured_edit_ab_{args.category}_{args.split}_{args.gallery}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"saved={output.resolve()}")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
