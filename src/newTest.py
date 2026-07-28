"""FashionIQ evaluation for DQU-CIR + Qwen structured text."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
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
    structured_weights = []
    predicted_gates = []
    dqu_text_weights = []
    residual_norms = []
    effective_residual_norms = []
    text_drifts = []
    field_entropies = []
    with torch.inference_mode():
        for batch in tqdm(
            list(_batched(queries, params.batch_size)),
            desc="query features",
            disable=not _show_progress(params),
        ):
            images = torch.stack([item["visual_query"] for item in batch]).to(device)
            raw_text = [item["textual_query"] for item in batch]
            structured_text = [item["structured_text"] for item in batch]
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
            if getattr(params, "disable_structured_text", False):
                mask.zero_()
            with _autocast(device):
                structured_fields = list(
                    zip(*[item["structured_fields"] for item in batch])
                )
                structured_field_mask = torch.tensor(
                    [item["structured_field_mask"] for item in batch],
                    device=device,
                    dtype=torch.bool,
                )
                features, diagnostics = model.extract_query(
                    raw_text,
                    structured_text,
                    images,
                    mask,
                    confidence,
                    structured_fields=structured_fields,
                    structured_field_mask=structured_field_mask,
                    return_diagnostics=True,
                )
            query_batches.append(features.float().cpu())
            structured_weights.append(
                diagnostics["structured_weight"].float().cpu()
            )
            predicted_gates.append(
                diagnostics["predicted_structured_gate"].float().cpu()
            )
            dqu_text_weights.append(
                diagnostics["dqu_text_weight"].float().cpu()
            )
            residual_norms.append(
                diagnostics["adapter_residual_norm"].float().cpu()
            )
            effective_residual_norms.append(
                diagnostics["effective_residual_norm"].float().cpu()
            )
            text_drifts.append(
                diagnostics["text_drift"].float().cpu()
            )
            field_entropies.append(
                diagnostics["field_attention_entropy"].float().cpu()
            )

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
    valid_confidences = [
        item["structured_confidence"]
        for item in queries
        if item["has_structured_text"]
    ]
    out = [
        (f"{category}_r{k}", float(np.mean(ranks_array <= k) * 100.0))
        for k in (1, 10, 50)
    ]
    out.extend(
        [
            (f"{category}_median_rank", float(np.median(ranks_array))),
            (f"{category}_mean_rank", float(np.mean(ranks_array))),
            (
                f"{category}_structured_weight",
                float(torch.cat(structured_weights).mean().item()),
            ),
            (
                f"{category}_predicted_structured_gate",
                float(torch.cat(predicted_gates).mean().item()),
            ),
            (
                f"{category}_adapter_residual_norm",
                float(torch.cat(residual_norms).mean().item()),
            ),
            (
                f"{category}_effective_residual_norm",
                float(torch.cat(effective_residual_norms).mean().item()),
            ),
            (
                f"{category}_text_drift",
                float(torch.cat(text_drifts).mean().item()),
            ),
            (
                f"{category}_field_attention_entropy",
                float(torch.cat(field_entropies).mean().item()),
            ),
            (
                f"{category}_dqu_text_weight",
                float(torch.cat(dqu_text_weights).mean().item()),
            ),
            (
                f"{category}_mean_qwen_confidence",
                float(np.mean(valid_confidences)) if valid_confidences else 0.0,
            ),
            (
                f"{category}_structured_coverage",
                float(
                    np.mean(
                        [item["has_structured_text"] for item in queries]
                    )
                ),
            ),
        ]
    )
    return out
