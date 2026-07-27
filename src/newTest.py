"""FashionIQ retrieval evaluation for the structured-description model."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _show_progress(params) -> bool:
    return getattr(params, "local_rank", -1) in (-1, 0)


def _batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def test(params, model, testset, category):
    """Compute FashionIQ R@1/R@10/R@50 over the configured gallery.

    The source image is removed from each ranking.  Query and gallery features
    are L2-normalized by the model, so their matrix product is cosine
    similarity.  ``fashioniq_split=original-split`` uses the official full
    validation gallery; ``val-split`` is only a smaller debugging gallery.
    """

    model.eval()
    device = next(model.parameters()).device
    queries = testset.test_queries
    eval_limit = int(getattr(params, "eval_limit", 0) or 0)
    if eval_limit > 0:
        queries = queries[:eval_limit]
    targets = testset.test_targets
    if not queries:
        raise ValueError("No FashionIQ evaluation queries are available")
    if not targets:
        raise ValueError("No FashionIQ gallery images are available")

    query_batches = []
    target_weights = []
    image_weights = []
    with torch.inference_mode():
        for batch in tqdm(
            list(_batched(queries, params.batch_size)),
            desc="query features",
            disable=not _show_progress(params),
        ):
            images = torch.stack([item["visual_query"] for item in batch]).to(device)
            raw_text = [item["textual_query"] for item in batch]
            target_text = [item["target_description"] for item in batch]
            mask = torch.tensor(
                [item["has_target_description"] for item in batch],
                device=device,
                dtype=torch.bool,
            )
            if getattr(params, "disable_target_description", False):
                mask.zero_()
            with _autocast(device):
                features, diagnostics = model.extract_query(
                    raw_text,
                    target_text,
                    images,
                    mask,
                    return_diagnostics=True,
                )
            query_batches.append(features.float().cpu())
            target_weights.append(diagnostics["target_weight"].float().cpu())
            image_weights.append(diagnostics["image_weight"].float().cpu())

        gallery_batches = []
        for batch in tqdm(
            list(_batched(targets, params.batch_size)),
            desc="gallery features",
            disable=not _show_progress(params),
        ):
            images = torch.stack([item["target_img_data"] for item in batch]).to(device)
            with _autocast(device):
                gallery_batches.append(model.extract_target(images).float().cpu())

    query_features = torch.cat(query_batches)
    gallery_features = torch.cat(gallery_batches)
    similarities = query_features @ gallery_features.t()

    gallery_index = {}
    for index, item in enumerate(targets):
        image_id = item["target_img_id"]
        if image_id in gallery_index:
            raise ValueError(f"Duplicate gallery id: {image_id}")
        gallery_index[image_id] = index

    target_positions = []
    ranks = []
    for row, query in enumerate(queries):
        source_id = query["source_img_id"]
        target_id = query["target_img_id"]
        if source_id not in gallery_index or target_id not in gallery_index:
            raise KeyError(
                f"Query references an id outside the gallery: source={source_id}, "
                f"target={target_id}"
            )
        similarities[row, gallery_index[source_id]] = -torch.inf
        order = torch.argsort(similarities[row], descending=True)
        target_position = gallery_index[target_id]
        rank = int(torch.nonzero(order == target_position, as_tuple=False)[0, 0]) + 1
        target_positions.append(target_position)
        ranks.append(rank)

    ranks_array = np.asarray(ranks)
    out = [
        (f"{category}_r{k}", float(np.mean(ranks_array <= k) * 100.0))
        for k in (1, 10, 50)
    ]
    out.extend(
        [
            (f"{category}_median_rank", float(np.median(ranks_array))),
            (f"{category}_mean_rank", float(np.mean(ranks_array))),
            (
                f"{category}_target_gate",
                float(torch.cat(target_weights).mean().item()),
            ),
            (
                f"{category}_image_gate",
                float(torch.cat(image_weights).mean().item()),
            ),
            (
                f"{category}_structured_coverage",
                float(
                    np.mean(
                        [item["has_target_description"] for item in queries]
                    )
                ),
            ),
        ]
    )
    return out
