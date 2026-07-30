"""Evaluate DQU and Qwen keyword-written image views with a frozen DQU model."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import newDataset
import newModel
from newTrain import default_structured_path, load_checkpoint, load_dqu_checkpoint


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
    parser.add_argument(
        "--adapter-checkpoint",
        help=(
            "Optional structured-text adapter checkpoint. When supplied, "
            "blend the two complete query views after DQU+adapter encoding."
        ),
    )
    parser.add_argument(
        "--max-structured-weight",
        type=float,
        help=(
            "Override the adapter checkpoint value. By default this is read "
            "from checkpoint config so evaluation reproduces newTrain.py."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--alphas",
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1",
        help="Qwen image-feature weights; 0 is DQU and 1 is Qwen-add.",
    )
    parser.add_argument("--output")
    return parser.parse_args()


def autocast(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def make_dataset(args, model, source):
    train_path = args.structured_train_path or default_structured_path(args, "train")
    val_path = args.structured_val_path or default_structured_path(args, "val")
    return newDataset.FashionIQ(
        path=args.fashioniq_path,
        category=args.dataset,
        transform=(model.preprocess_train, model.preprocess_val),
        split=args.fashioniq_split,
        structured_train_path=train_path,
        structured_val_path=val_path,
        use_written_image=True,
        written_keyword_source=source,
    )


def encode_queries(model, dqu_queries, qwen_queries, batch_size, device):
    text_features = []
    dqu_image_features = []
    qwen_image_features = []
    if len(dqu_queries) != len(qwen_queries):
        raise ValueError("DQU and Qwen query counts differ")
    with torch.inference_mode():
        paired = list(zip(dqu_queries, qwen_queries))
        for batch in tqdm(list(batches(paired, batch_size)), desc="query views"):
            dqu_batch, qwen_batch = zip(*batch)
            for left, right in zip(dqu_batch, qwen_batch):
                identity_left = (
                    left["source_img_id"],
                    left["target_img_id"],
                    left["textual_query"],
                )
                identity_right = (
                    right["source_img_id"],
                    right["target_img_id"],
                    right["textual_query"],
                )
                if identity_left != identity_right:
                    raise ValueError("DQU and Qwen queries are not aligned")
            texts = [item["textual_query"] for item in dqu_batch]
            dqu_images = torch.stack(
                [item["visual_query"] for item in dqu_batch]
            ).to(device)
            qwen_images = torch.stack(
                [item["visual_query"] for item in qwen_batch]
            ).to(device)
            with autocast(device):
                text_features.append(
                    F.normalize(model.extract_text_fea(texts), dim=-1).float().cpu()
                )
                dqu_image_features.append(
                    F.normalize(model.extract_img_fea(dqu_images), dim=-1)
                    .float()
                    .cpu()
                )
                qwen_image_features.append(
                    F.normalize(model.extract_img_fea(qwen_images), dim=-1)
                    .float()
                    .cpu()
                )
    return (
        torch.cat(text_features),
        torch.cat(dqu_image_features),
        torch.cat(qwen_image_features),
    )


def encode_gallery(model, targets, batch_size, device):
    features = []
    with torch.inference_mode():
        for batch in tqdm(list(batches(targets, batch_size)), desc="gallery"):
            images = torch.stack([item["target_img_data"] for item in batch]).to(
                device
            )
            with autocast(device):
                features.append(model.extract_target(images).float().cpu())
    return torch.cat(features)


def encode_complete_queries(model, queries, batch_size, device):
    features = []
    with torch.inference_mode():
        for batch in tqdm(list(batches(queries, batch_size)), desc="full queries"):
            images = torch.stack([item["visual_query"] for item in batch]).to(device)
            mask = torch.tensor(
                [item["has_structured_text"] for item in batch],
                device=device,
                dtype=torch.bool,
            )
            confidence = torch.tensor(
                [item["structured_confidence"] for item in batch],
                device=device,
                dtype=torch.float32,
            )
            field_mask = torch.tensor(
                [item["structured_field_mask"] for item in batch],
                device=device,
                dtype=torch.bool,
            )
            quality = torch.tensor(
                [item["structured_quality_features"] for item in batch],
                device=device,
                dtype=torch.float32,
            )
            fields = list(zip(*[item["structured_fields"] for item in batch]))
            with autocast(device):
                query = model.extract_query(
                    [item["textual_query"] for item in batch],
                    [item["structured_text"] for item in batch],
                    images,
                    mask,
                    confidence,
                    structured_fields=fields,
                    structured_field_mask=field_mask,
                    structured_quality_features=quality,
                )
            features.append(query.float().cpu())
    return torch.cat(features)


def recalls(query_features, gallery_features, queries, targets):
    similarities = query_features @ gallery_features.t()
    gallery_index = {
        item["target_img_id"]: index for index, item in enumerate(targets)
    }
    ranks = []
    for row, query in enumerate(queries):
        similarities[row, gallery_index[query["source_img_id"]]] = -torch.inf
        order = torch.argsort(similarities[row], descending=True)
        target_position = gallery_index[query["target_img_id"]]
        rank = int(torch.nonzero(order == target_position)[0, 0]) + 1
        ranks.append(rank)
    ranks = np.asarray(ranks)
    return {
        "r1": float(np.mean(ranks <= 1) * 100.0),
        "r10": float(np.mean(ranks <= 10) * 100.0),
        "r50": float(np.mean(ranks <= 50) * 100.0),
        "mean_rank": float(np.mean(ranks)),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ViT-H/14 evaluation requires CUDA")
    device = torch.device("cuda")
    adapter_config = {}
    if args.adapter_checkpoint:
        adapter_payload = torch.load(args.adapter_checkpoint, map_location="cpu")
        adapter_config = adapter_payload.get("config", {})
    max_structured_weight = args.max_structured_weight
    if max_structured_weight is None:
        max_structured_weight = float(
            adapter_config.get("max_structured_weight", 0.25)
        )
    model = newModel.DQU_CIR(
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        clip_checkpoint=args.clip_checkpoint,
        clip_cache_dir=args.clip_cache_dir,
        freeze_clip=True,
        adapter_rank=int(adapter_config.get("adapter_rank", 256)),
        max_structured_weight=max_structured_weight,
        max_residual_norm=float(adapter_config.get("max_residual_norm", 1.0)),
    ).to(device)
    load_dqu_checkpoint(args.dqu_checkpoint, model)
    if args.adapter_checkpoint:
        load_checkpoint(args.adapter_checkpoint, model)
    model.freeze_dqu_backbone()
    model.eval()

    dqu_dataset = make_dataset(args, model, "dqu")
    qwen_dataset = make_dataset(args, model, "qwen-add")
    gallery = encode_gallery(
        model, dqu_dataset.test_targets, args.batch_size, device
    )
    if args.adapter_checkpoint:
        dqu_query = encode_complete_queries(
            model, dqu_dataset.test_queries, args.batch_size, device
        )
        qwen_query = encode_complete_queries(
            model, qwen_dataset.test_queries, args.batch_size, device
        )
    else:
        text, dqu_image, qwen_image = encode_queries(
            model,
            dqu_dataset.test_queries,
            qwen_dataset.test_queries,
            args.batch_size,
            device,
        )

    alphas = [float(value) for value in args.alphas.split(",")]
    results = []
    with torch.inference_mode():
        for alpha in alphas:
            if args.adapter_checkpoint:
                query_features = F.normalize(
                    (1.0 - alpha) * dqu_query + alpha * qwen_query, dim=-1
                )
            else:
                image = F.normalize(
                    (1.0 - alpha) * dqu_image + alpha * qwen_image, dim=-1
                ).to(device)
                query_batches = []
                for start in range(0, len(text), args.batch_size):
                    with autocast(device):
                        query, _ = model._dqu_fusion(
                            text[start : start + args.batch_size].to(device),
                            image[start : start + args.batch_size],
                        )
                    query_batches.append(query.float().cpu())
                query_features = torch.cat(query_batches)
            metrics = recalls(
                query_features,
                gallery,
                dqu_dataset.test_queries,
                dqu_dataset.test_targets,
            )
            metrics["alpha"] = alpha
            metrics["score"] = metrics["r10"] + metrics["r50"]
            results.append(metrics)
            print(json.dumps(metrics, sort_keys=True))

    blend_level = "complete query" if args.adapter_checkpoint else "image feature"
    payload = {
        "description": (
            f"{blend_level} blend: alpha=0 DQU-written image; "
            "alpha=1 Qwen-add-written image"
        ),
        "results": results,
    }
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
