"""Evaluate a bounded generated-image residual on top of the best dual view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from evaluate_image_keyword_routes import (
    autocast,
    batches,
    encode_complete_queries,
    encode_gallery,
    make_dataset,
)
from newTrain import load_checkpoint, load_dqu_checkpoint
import newModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dress")
    parser.add_argument("--fashioniq-path", default="../data/FashionIQ")
    parser.add_argument("--fashioniq-split", default="original-split")
    parser.add_argument("--structured-train-path")
    parser.add_argument("--structured-val-path")
    parser.add_argument("--clip-model", default="ViT-H-14")
    parser.add_argument("--clip-pretrained", default="laion2B-s32B-b79K")
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--clip-cache-dir", default="../model_cache")
    parser.add_argument("--dqu-checkpoint", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dual-alpha", type=float, default=0.5)
    parser.add_argument("--lambdas", default="0,0.05,0.1,0.15,0.2")
    parser.add_argument(
        "--generated-encoding",
        choices=("complete", "target"),
        default="complete",
        help=(
            "complete applies the relative caption to the generated image; "
            "target encodes a text-generated target hypothesis as an image only."
        ),
    )
    parser.add_argument(
        "--fusion-modes",
        default="residual,direct",
        help=(
            "residual adds generated-minus-plain query; direct interpolates "
            "the complete generated query as a third view."
        ),
    )
    parser.add_argument("--shuffle-seed", type=int, default=20260730)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def ranks(query_features, gallery_features, queries, targets):
    similarities = query_features @ gallery_features.t()
    gallery_index = {
        item["target_img_id"]: index for index, item in enumerate(targets)
    }
    output = []
    for row, query in enumerate(queries):
        similarities[row, gallery_index[query["source_img_id"]]] = -torch.inf
        order = torch.argsort(similarities[row], descending=True)
        target_index = gallery_index[query["target_img_id"]]
        output.append(int(torch.nonzero(order == target_index)[0, 0]) + 1)
    return np.asarray(output)


def metrics(values):
    return {
        "r1": float(np.mean(values <= 1) * 100),
        "r10": float(np.mean(values <= 10) * 100),
        "r50": float(np.mean(values <= 50) * 100),
        "mean_rank": float(np.mean(values)),
    }


def generated_items(plain_queries, manifest, transform):
    records = {
        (item["candidate"], item["target"]): item
        for item in manifest["records"]
    }
    items = []
    indices = []
    confidence = []
    for index, query in enumerate(plain_queries):
        key = (query["source_img_id"], query["target_img_id"])
        record = records.get(key)
        if not record:
            continue
        with Image.open(record["generated_path"]) as handle:
            image = handle.convert("RGB")
            pixels = np.asarray(image)
            # Diffusers' safety checker can intentionally return a black image.
            # Treat that as a missing view so it contributes exactly zero
            # residual instead of a large, meaningless feature displacement.
            if pixels.mean() < 5 or pixels.std() < 2:
                continue
            visual = transform(image)
        item = dict(query)
        item["visual_query"] = visual
        items.append(item)
        indices.append(index)
        confidence.append(float(record.get("confidence", 1.0)))
    return items, np.asarray(indices), torch.tensor(confidence).float()


def encode_generated_targets(model, items, batch_size, device):
    features = []
    with torch.inference_mode():
        for batch in batches(items, batch_size):
            images = torch.stack(
                [item["visual_query"] for item in batch]
            ).to(device)
            with autocast(device):
                features.append(model.extract_target(images).float().cpu())
    return torch.cat(features)


def main():
    args = parse_args()
    device = torch.device("cuda")
    payload = torch.load(args.adapter_checkpoint, map_location="cpu")
    config = payload.get("config", {})
    model = newModel.DQU_CIR(
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        clip_checkpoint=args.clip_checkpoint,
        clip_cache_dir=args.clip_cache_dir,
        freeze_clip=True,
        adapter_rank=int(config.get("adapter_rank", 256)),
        max_structured_weight=float(config.get("max_structured_weight", 0.25)),
        max_residual_norm=float(config.get("max_residual_norm", 1.0)),
    ).to(device)
    load_dqu_checkpoint(args.dqu_checkpoint, model)
    load_checkpoint(args.adapter_checkpoint, model)
    model.freeze_dqu_backbone()
    model.eval()

    dqu = make_dataset(args, model, "dqu")
    qwen = make_dataset(args, model, "qwen-add")
    plain = make_dataset(args, model, "dqu")
    plain.use_written_image = False
    plain.test_queries, plain.test_targets = plain.get_test_data()
    gallery = encode_gallery(model, dqu.test_targets, args.batch_size, device)
    dqu_query = encode_complete_queries(
        model, dqu.test_queries, args.batch_size, device
    )
    qwen_query = encode_complete_queries(
        model, qwen.test_queries, args.batch_size, device
    )
    plain_query = encode_complete_queries(
        model, plain.test_queries, args.batch_size, device
    )
    dual = F.normalize(
        (1 - args.dual_alpha) * dqu_query + args.dual_alpha * qwen_query, dim=-1
    )

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    generated, selected, confidence = generated_items(
        plain.test_queries, manifest, model.preprocess_val
    )
    if not len(generated):
        raise ValueError("No manifest records matched validation queries")
    if args.generated_encoding == "complete":
        generated_query = encode_complete_queries(
            model, generated, args.batch_size, device
        )
    else:
        generated_query = encode_generated_targets(
            model, generated, args.batch_size, device
        )
    delta = generated_query - plain_query[selected]
    delta = delta * confidence[:, None].clamp(0, 1)
    permutation = torch.randperm(
        len(delta), generator=torch.Generator().manual_seed(args.shuffle_seed)
    )

    baseline_ranks = ranks(
        dual, gallery, dqu.test_queries, dqu.test_targets
    )
    results = []
    fusion_modes = [
        value.strip() for value in args.fusion_modes.split(",") if value.strip()
    ]
    invalid_modes = set(fusion_modes) - {"residual", "direct"}
    if invalid_modes:
        raise ValueError(f"Unknown fusion modes: {sorted(invalid_modes)}")
    for mode in fusion_modes:
        for lam in [float(value) for value in args.lambdas.split(",")]:
            for control, order in (
                ("generated", torch.arange(len(selected))),
                ("shuffled", permutation),
            ):
                query = dual.clone()
                if mode == "residual":
                    update = delta[order]
                    query[selected] = F.normalize(
                        dual[selected] + lam * update, dim=-1
                    )
                else:
                    weight = (
                        lam * confidence[order].clamp(0, 1)
                    )[:, None]
                    generated_view = generated_query[order]
                    query[selected] = F.normalize(
                        (1.0 - weight) * dual[selected]
                        + weight * generated_view,
                        dim=-1,
                    )
                current_ranks = ranks(
                    query, gallery, dqu.test_queries, dqu.test_targets
                )
                selected_before = baseline_ranks[selected]
                selected_after = current_ranks[selected]
                row = {
                    "fusion_mode": mode,
                    "lambda": lam,
                    "control": control,
                    "all": metrics(current_ranks),
                    "selected": metrics(selected_after),
                    "selected_count": int(len(selected)),
                    "rank_wins": int(np.sum(selected_after < selected_before)),
                    "rank_ties": int(
                        np.sum(selected_after == selected_before)
                    ),
                    "rank_losses": int(
                        np.sum(selected_after > selected_before)
                    ),
                    "mean_rank_delta": float(
                        np.mean(selected_after - selected_before)
                    ),
                }
                row["all"]["score"] = (
                    row["all"]["r10"] + row["all"]["r50"]
                )
                row["selected"]["score"] = (
                    row["selected"]["r10"] + row["selected"]["r50"]
                )
                results.append(row)
                print(json.dumps(row, sort_keys=True))
    output = {
        "description": (
            "Fixed-alpha dual written-image query plus either a confidence-"
            "gated generated residual or direct third-view interpolation; "
            "shuffled generated views are the negative control."
        ),
        "generated_encoding": args.generated_encoding,
        "baseline": {
            "all": metrics(baseline_ranks),
            "selected": metrics(baseline_ranks[selected]),
        },
        "results": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
