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
from newTrain import default_structured_path, load_dqu_checkpoint


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
    model = newModel.DQU_CIR(
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        clip_checkpoint=args.clip_checkpoint,
        clip_cache_dir=args.clip_cache_dir,
        freeze_clip=True,
    ).to(device)
    load_dqu_checkpoint(args.dqu_checkpoint, model)
    model.freeze_dqu_backbone()
    model.eval()

    dqu_dataset = make_dataset(args, model, "dqu")
    qwen_dataset = make_dataset(args, model, "qwen-add")
    text, dqu_image, qwen_image = encode_queries(
        model,
        dqu_dataset.test_queries,
        qwen_dataset.test_queries,
        args.batch_size,
        device,
    )
    gallery = encode_gallery(
        model, dqu_dataset.test_targets, args.batch_size, device
    )

    alphas = [float(value) for value in args.alphas.split(",")]
    results = []
    with torch.inference_mode():
        for alpha in alphas:
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
            metrics = recalls(
                torch.cat(query_batches),
                gallery,
                dqu_dataset.test_queries,
                dqu_dataset.test_targets,
            )
            metrics["alpha"] = alpha
            metrics["score"] = metrics["r10"] + metrics["r50"]
            results.append(metrics)
            print(json.dumps(metrics, sort_keys=True))

    payload = {
        "description": "alpha=0 DQU-written image; alpha=1 Qwen-add-written image",
        "results": results,
    }
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
