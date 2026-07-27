"""FashionIQ retrieval evaluation under the official DQU-CIR protocol."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def test(params, model, testset, category):
    """Return R@1/R@10/R@50, ranks, and the learned text fusion weight."""
    model.eval()
    device = next(model.parameters()).device
    queries = testset.test_queries
    targets = testset.test_targets
    if not queries or not targets:
        raise ValueError("FashionIQ evaluation requires non-empty queries/gallery")

    query_features = []
    scalers = []
    gallery_features = []
    with torch.inference_mode():
        for batch in tqdm(
            list(_batched(queries, params.batch_size)),
            desc="DQU query features",
        ):
            visual = torch.stack([item["visual_query"] for item in batch]).to(
                device, non_blocking=True
            )
            text = [item["textual_query"] for item in batch]
            with _autocast(device):
                features, scaler = model.extract_query(
                    text, visual, return_scaler=True
                )
            query_features.append(features.float().cpu())
            scalers.append(scaler.float().cpu())

        for batch in tqdm(
            list(_batched(targets, params.batch_size)),
            desc="DQU gallery features",
        ):
            images = torch.stack([item["target_img_data"] for item in batch]).to(
                device, non_blocking=True
            )
            with _autocast(device):
                gallery_features.append(model.extract_target(images).float().cpu())

    query_features = torch.cat(query_features)
    gallery_features = torch.cat(gallery_features)
    similarities = query_features @ gallery_features.t()
    gallery_index = {
        item["target_img_id"]: index for index, item in enumerate(targets)
    }
    if len(gallery_index) != len(targets):
        raise ValueError("Duplicate image ids in FashionIQ gallery")

    ranks = []
    for row, query in enumerate(queries):
        source_position = gallery_index[query["source_img_id"]]
        target_position = gallery_index[query["target_img_id"]]
        similarities[row, source_position] = -torch.inf
        order = torch.argsort(similarities[row], descending=True)
        rank = int(torch.nonzero(order == target_position, as_tuple=False)[0]) + 1
        ranks.append(rank)

    ranks = np.asarray(ranks)
    metrics = [
        (f"{category}_r{k}", float(np.mean(ranks <= k) * 100.0))
        for k in (1, 10, 50)
    ]
    metrics.extend(
        [
            (f"{category}_median_rank", float(np.median(ranks))),
            (f"{category}_mean_rank", float(np.mean(ranks))),
            (f"{category}_text_weight", float(torch.cat(scalers).mean())),
        ]
    )
    return metrics
