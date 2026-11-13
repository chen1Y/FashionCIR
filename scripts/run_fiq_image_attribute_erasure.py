"""Paired FashionIQ evaluation of the learned image attribute eraser."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch
from torch.nn.utils import parametrize

from fiq_conflict_context_040 import ordinary_text
from fiq_generated_text_only_040de import GeneratedCaptionText
from fiq_native_caption_math_031 import metrics, transitions
from fiq_pixel_free_038 import PixelFreeVisual
from fiq_vit_lora_039 import attach_lora
from run_029d_freeze_patch import ranks_for
from run_029d_openclip_036 import from_public_clip, gpu_snapshot, official_bank
from run_fiq_native_caption_031 import identity, read, sha, write
from run_fiq_vit_lora_039 import PREFIX_SHAPE, replay_bank, score_bank
from run_fiq_vit_lora_039a_earlystop import bind_full_texts


EXPERIMENT = "图像属性擦除验证实验"
GPU_INDEX = 0
GPU_BUDGET_MIB = 8192
SPLIT_SEED = "fashioniq-attribute-erasure-probe-v1"
PROBE_CAP_PER_CATEGORY_CLASS = 128
MIN_FIT_PER_CLASS = 30
MIN_HOLDOUT_PER_CLASS = 12
MIN_HOLDOUT_AUC = 0.70

PATTERNS = {
    "white": r"\bwhite\b",
    "black": r"\bblack\b",
    "blue": r"\bblue\b",
    "long sleeves": r"\blong[ -]sleeve(?:s|d)?\b",
    "short sleeves": r"\bshort[ -]sleeve(?:s|d)?\b",
    "crew neck": r"\b(?:crew[ -]?neck|round[ -]neck(?:line)?)\b",
    "v-neck": r"\bv[ -]?neck(?:line)?\b",
    "floral pattern": r"\bfloral\b",
    "striped pattern": r"\b(?:stripes?|striped)\b",
}
GROUPS = (
    ("white", "black", "blue"),
    ("long sleeves", "short sleeves"),
    ("crew neck", "v-neck"),
    ("floral pattern", "striped pattern"),
)
COMPARATIVE = re.compile(
    r"\b(?:lighter|darker|brighter|longer|shorter|wider|narrower|more|less)\b", re.I
)


def attributes(text):
    return {attr for attr, pattern in PATTERNS.items() if re.search(pattern, text, re.I)}


def weak_labels(caption):
    """Use explicit alternatives for negatives and abstain on ambiguous/omitted groups."""
    seen = attributes(caption)
    labels = {}
    for group in GROUPS:
        present = seen.intersection(group)
        if len(present) == 1:
            labels.update({attr: int(attr in present) for attr in group})
    return labels


def digest_order(digest):
    return hashlib.sha256(f"{SPLIT_SEED}:{digest}".encode()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def load_probe_candidates(source, data, plan):
    categories = plan["categories"]
    val_digests = set()
    training = {}
    for category in categories:
        metadata = source / "metadata" / f"{category}.json"
        spec = plan["inputs"][category]
        if sha(metadata) != spec["metadata_sha256"]:
            raise ValueError(f"Training metadata differs for {category}")
        meta = read(metadata)
        val_meta, val_hashes, _ = official_bank(category)
        val_digests.update(val_hashes[name] for name in val_meta["ids"])
        training[category] = meta, spec["train_prefix"]

    seen, entries, by_category = set(), {}, {}
    for category, (meta, prefix) in training.items():
        captions_file = data / "captions" / f"image_captions_{category}_train.json"
        captions = read(captions_file)
        ids = list(dict.fromkeys(meta["banks"]["fit"] + meta["banks"]["dev"]))
        path = Path(prefix["path"])
        if path.stat().st_size != prefix["bytes"]:
            raise ValueError(f"Training prefix byte count differs for {category}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != (len(ids), *PREFIX_SHAPE) or array.dtype != np.uint16:
            raise ValueError(f"Training prefix layout differs for {category}")
        by_category[category] = dict(cache={"prefix": array}, captions_sha256=sha(captions_file),
                                     prefix=str(path), registered_prefix_sha256=prefix["sha256"])
        groups = {}
        for index, name in enumerate(ids):
            if name not in captions:
                continue
            digest = meta["image_sha256"][name]
            if digest in val_digests or digest in seen:
                continue
            labels = weak_labels(captions[name])
            if not labels:
                continue
            seen.add(digest)
            order = digest_order(digest)
            split = "holdout" if int(order[:8], 16) % 4 == 0 else "fit"
            for attr, label in labels.items():
                groups.setdefault((attr, split, label), []).append((order, index))
        for (attr, split, label), candidates in groups.items():
            selected = sorted(candidates)[:PROBE_CAP_PER_CATEGORY_CLASS]
            entries.setdefault((attr, split, label), []).extend((category, index) for _, index in selected)
    return entries, by_category


def auc_for(scores_pos, scores_neg):
    pair = scores_pos[:, None] - scores_neg[None, :]
    return float(((pair > 0).float() + .5 * (pair == 0)).mean())


def fit_probes(model, entries, by_category, report):
    needed = {cat: set() for cat in by_category}
    for pairs in entries.values():
        for category, index in pairs:
            needed[category].add(index)
    features = {}
    for category, selected in needed.items():
        if not selected:
            continue
        positions = sorted(selected)
        encoded = score_bank(model.clip.visual, by_category[category]["cache"],
                             lambda done, total: report("probe_features", category, done, total),
                             np.asarray(positions)).cpu()
        features.update({(category, index): feature for index, feature in zip(positions, encoded)})

    probes, audit = {}, {}
    for attr in PATTERNS:
        keys = {(split, label): entries.get((attr, split, label), [])
                for split in ("fit", "holdout") for label in (0, 1)}
        sizes = {f"{split}_{label}": len(items) for (split, label), items in keys.items()}
        status = audit[attr] = dict(counts=sizes, status="insufficient_examples")
        if (min(sizes["fit_0"], sizes["fit_1"]) < MIN_FIT_PER_CLASS
                or min(sizes["holdout_0"], sizes["holdout_1"]) < MIN_HOLDOUT_PER_CLASS):
            continue
        vectors = {key: torch.stack([features[pair] for pair in pairs]) for key, pairs in keys.items()}
        negative, positive = vectors["fit", 0], vectors["fit", 1]
        mean = .5 * (negative.mean(0) + positive.mean(0))
        negative, positive = negative - mean, positive - mean
        covariance = .5 * (negative.T @ negative / len(negative) + positive.T @ positive / len(positive))
        direction = .5 * (positive.mean(0) - negative.mean(0))
        weight = torch.linalg.solve(covariance + .01 * torch.eye(len(mean)), direction)
        bias = -float(mean @ weight)
        pos = vectors["holdout", 1] @ weight + bias
        neg = vectors["holdout", 0] @ weight + bias
        auc = auc_for(pos, neg)
        status.update(holdout_auc=auc,
                      holdout_balanced_accuracy=.5 * (float((pos >= 0).float().mean())
                                                       + float((neg < 0).float().mean())),
                      status="qualified" if auc >= MIN_HOLDOUT_AUC else "below_holdout_auc")
        if auc >= MIN_HOLDOUT_AUC:
            probes[attr] = (weight, bias)
    return probes, audit


def load_best(model, visual, path):
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    state = checkpoint["state"]
    params = {name: parameter for name, parameter in model.clip.named_parameters() if parameter.requires_grad}
    if set(params) != set(state["blocks"]):
        raise ValueError("Saved 040E parameters differ from the reconstructed architecture")
    with torch.no_grad():
        for name, parameter in params.items():
            parameter.copy_(state["blocks"][name])
    model.image_fusion.load_state_dict(state["image_fusion"], strict=True)
    visual.load_state_dict(checkpoint["visual"], strict=True)
    model.eval()
    visual.eval()
    return int(checkpoint["epoch"])


def compare_category(model, visual, bank, original, captions, probes, category, report):
    gallery = bank["gallery"]
    ids, inverse = bank["references"].unique(return_inverse=True)
    references = score_bank(model.clip.visual, bank,
                            lambda done, total: report("validation_reference", category, done, total),
                            ids.cpu().numpy())[inverse]
    before_batches, after_batches = [], []
    alignment = []
    probe_rows = []
    attrs = list(probes)
    if attrs:
        weights = torch.stack([probes[attr][0] for attr in attrs])
        biases = torch.tensor([probes[attr][1] for attr in attrs])
    for start in range(0, len(references), 8):
        end = min(start + 8, len(references))
        rows = bank["rows"][start:end]
        refs, drop = references[start:end], bank["drop"][start:end]
        active = bank["drop_active"][start:end]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            text = ordinary_text(model, [row["main_context_text"] for row in rows])
            clean, _ = visual(refs, text, drop, active)
            query_after, _ = model.image_fusion(text, clean)
            query_before, _ = model.image_fusion(text, refs)
            with torch.autocast("cuda", enabled=False):
                after_batches.append((query_after.float() @ gallery.T).cpu())
                before_batches.append((query_before.float() @ gallery.T).cpu())
        before_cos = (refs * drop).sum(-1).cpu().tolist()
        after_cos = (clean * drop).sum(-1).cpu().tolist()
        shifts = (clean - refs).norm(dim=-1).cpu().tolist()
        active = active.cpu().tolist()
        if attrs:
            score_before = (refs.cpu() @ weights.T + biases).tolist()
            score_after = (clean.cpu() @ weights.T + biases).tolist()
        for i, row in enumerate(rows):
            index = row["source_index"]
            comparative = bool(COMPARATIVE.search(" ".join(original[index]["captions"])))
            drop_text = row["drop_attrs"]
            alignment.append(dict(category=category, source_index=index, reference=row["candidate"],
                                  target=row["target"], comparative=comparative, erasure_active=active[i],
                                  drop_attrs=drop_text, kept_context=row["kept_context"],
                                  drop_cos_before=before_cos[i], drop_cos_after=after_cos[i],
                                  visual_l2_shift=shifts[i]))
            if not active[i] or not attrs:
                continue
            erased = attributes(drop_text)
            kept = attributes(row["kept_context"]) - erased
            independent = weak_labels(captions.get(row["candidate"], ""))
            for j, attr in enumerate(attrs):
                role = "drop" if attr in erased else "keep" if attr in kept else None
                if role is None:
                    continue
                probe_rows.append(dict(category=category, source_index=index, reference=row["candidate"],
                                       attribute=attr, role=role, comparative=comparative,
                                       native_caption_confirms=independent.get(attr) == 1,
                                       score_before=score_before[i][j], score_after=score_after[i][j],
                                       score_delta=score_after[i][j] - score_before[i][j]))
        if start % 256 == 0 or end == len(references):
            report("paired_scoring", category, end, len(references))

    before, after = torch.cat(before_batches), torch.cat(after_batches)
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise FloatingPointError("Nonfinite paired retrieval scores")
    ranks_before, ranks_after = ranks_for(before, bank), ranks_for(after, bank)
    for row, off, on in zip(alignment, ranks_before, ranks_after):
        row.update(rank_off=off, rank_on=on)
        if not row["erasure_active"] and (off != on or row["visual_l2_shift"] != 0):
            raise ValueError("Erasure changed an inactive query")
    return alignment, probe_rows, ranks_before, ranks_after


def summarize(rows):
    result = {}
    for name, group in (
        ("all", rows),
        ("active", [row for row in rows if row["erasure_active"]]),
        ("comparative_active", [row for row in rows if row["erasure_active"] and row["comparative"]]),
        ("noncomparative_active", [row for row in rows if row["erasure_active"] and not row["comparative"]]),
    ):
        if not group:
            continue
        off, on = ([row[key] for row in group] for key in ("rank_off", "rank_on"))
        result[name] = dict(queries=len(group), off=metrics(off), on=metrics(on),
                            transitions={str(k): transitions(on, off, k) for k in (1, 10, 50)})
        if name != "all":
            changes = [row["drop_cos_after"] - row["drop_cos_before"] for row in group]
            result[name].update(mean_drop_cos_delta=sum(changes) / len(changes),
                                decreased=sum(x < -1e-6 for x in changes),
                                unchanged=sum(abs(x) <= 1e-6 for x in changes),
                                increased=sum(x > 1e-6 for x in changes))
    return result


def summarize_probes(rows):
    out = []
    for (attr, role, confirmed) in sorted({
        (row["attribute"], row["role"], row["native_caption_confirms"]) for row in rows
    }):
        group = [row for row in rows if (row["attribute"], row["role"], row["native_caption_confirms"])
                 == (attr, role, confirmed)]
        values = [row["score_delta"] for row in group]
        out.append(dict(attribute=attr, role=role, native_caption_confirms=confirmed, queries=len(values),
                        mean_score_delta=sum(values) / len(values), median_score_delta=float(np.median(values)),
                        fraction_decreased=sum(x < -1e-6 for x in values) / len(values)))
    return out


def run(root, source, data, gpu):
    state_path = root / "state.json"
    if state_path.exists():
        raise FileExistsError(state_path)
    plan = read(source / "plan.json")
    best = read(source / "best.json")
    state = dict(experiment=EXPERIMENT, status="running", identity=identity(), gpu=gpu, gpu_index=GPU_INDEX)

    def report(stage, category=None, done=0, total=0):
        state.update(stage=stage, category=category, done=done, total=total)
        write(state_path, state)
        print(stage, category, done, total, flush=True)

    try:
        report("verify_sources")
        if plan["arms"] != ["040E"] or not plan.get("full_caption") or plan["full_caption"].get("offsets"):
            raise ValueError("Expected the completed full Flash caption experiment")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
            raise ValueError("GPU UUID does not match the launch environment")
        for name, digest in plan["source_sha256"].items():
            if sha(source / "code" / name) != digest:
                raise ValueError(f"Archived source digest changed: {name}")
        text_manifest = source / "texts" / "inventory.json"
        if sha(text_manifest) != plan["text_inventory_sha256"]:
            raise ValueError("Flash text manifest digest changed")
        inventory = read(text_manifest)["splits"]
        checkpoint = source / best["weight"]
        if sha(checkpoint) != best["weight_sha256"]:
            raise ValueError("040E best checkpoint digest changed")
        archived = read(source / best["validation"])["rows"]
        write(root / "plan.json", dict(experiment=EXPERIMENT, source=str(source), best=best, gpu=gpu,
                                      gpu_budget_mib=GPU_BUDGET_MIB, original_plan_sha256=sha(source / "plan.json"),
                                      source_sha256=sha(Path(__file__)), api_calls=0, categories=plan["categories"],
                                      probe_labels="native FashionIQ BLIP-2 train captions, explicit alternatives",
                                      split_seed=SPLIT_SEED, probe_groups=GROUPS, min_fit=MIN_FIT_PER_CLASS,
                                      min_holdout=MIN_HOLDOUT_PER_CLASS, min_holdout_auc=MIN_HOLDOUT_AUC))
        report("check_gpu")
        for snapshot in range(2):
            card, _ = gpu_snapshot(gpu, GPU_INDEX)
            if card["free_mib"] < GPU_BUDGET_MIB + 2048:
                raise RuntimeError(f"GPU 0 free memory below {GPU_BUDGET_MIB + 2048} MiB")
            if snapshot == 0:
                time.sleep(5)
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(GPU_BUDGET_MIB * (1 << 20)
                                                   / torch.cuda.get_device_properties(0).total_memory)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        report("load_public_openclip")
        base, transform = from_public_clip(source / "weights" / "open_clip_pytorch_model.bin")
        model = GeneratedCaptionText(base, full_text=True)
        del base

        banks = {}
        for category in plan["categories"]:
            report("validation_prefix", category)
            val, hashes, _ = official_bank(category)
            rows, audit = bind_full_texts(source, inventory[f"{category}_val"], category, "val",
                                         val["rows"], hashes)
            if len(rows) != len(archived[category]["ranks"]):
                raise ValueError(f"Archived official rank count differs for {category}")
            bank = replay_bank(model, transform, category, rows, val["ids"], hashes,
                               plan["inputs"][category]["val_prefix"],
                               lambda stage, done, total: report(stage, category, done, total),
                               split="val", include_teacher=False)
            banks[category] = bank, audit
        # Gallery and drop features must be generated with the original frozen public encoder.
        list(attach_lora(model.clip.visual))
        visual = PixelFreeVisual(1024, True, False).cuda()
        epoch = load_best(model, visual, checkpoint)
        if epoch != best["epoch"]:
            raise ValueError("Checkpoint epoch differs from the archived best epoch")
        gamma = float(visual.eraser.gamma.detach().clamp(0., 1.5))
        report("probe_labels")
        entries, by_category = load_probe_candidates(source, data, plan)
        probes, probe_audit = fit_probes(model, entries, by_category, report)
        write(root / "probe_audit.json", dict(probes=probe_audit, qualified=list(probes),
              provenance={cat: {k: v for k, v in spec.items() if k != "cache"}
                          for cat, spec in by_category.items()}))
        del entries, by_category

        results, all_alignment, all_probe_rows = {}, [], []
        for category, (bank, audit) in banks.items():
            original = read(data / "captions" / f"cap.{category}.val.json")
            captions = read(data / "captions" / f"image_captions_{category}_val.json")
            rows, probe_rows, _, on = compare_category(model, visual, bank, original, captions, probes,
                                                       category, report)
            write_csv(root / f"{category}_per_query.csv", rows)
            write_csv(root / f"{category}_probe_scores.csv", probe_rows)
            reference_ranks = archived[category]["ranks"]
            results[category] = dict(gallery=len(bank["gallery"]), text_status=audit, metrics=summarize(rows),
                                     archived_rank_matches=sum(a == b for a, b in zip(on, reference_ranks)),
                                     archived_queries=len(reference_ranks))
            all_alignment.extend(rows)
            all_probe_rows.extend(probe_rows)
        write_csv(root / "probe_summary.csv", summarize_probes(all_probe_rows))
        macro = {arm: {f"r{k}": sum(results[cat]["metrics"]["all"][arm][f"r{k}"]
                                    for cat in results) / len(results) for k in (1, 10, 50)}
                 for arm in ("off", "on")}
        write(root / "summary.json", dict(experiment=EXPERIMENT, epoch=epoch, gamma=gamma,
                                          results=results, macro=macro, pooled=summarize(all_alignment),
                                          qualified_probes=list(probes), probe_rows=len(all_probe_rows),
                                          probe_summary=summarize_probes(all_probe_rows), api_calls=0))
        state.update(status="complete", stage="complete", category=None)
        write(state_path, state)
    except BaseException as exc:
        state.update(status="failed", error=type(exc).__name__, detail=str(exc))
        write(state_path, state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    run(args.root, args.source, args.data, args.gpu)
