"""Rerank the native DQU-CIR Top-100 using a frozen Flash text residual."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import cv2
import torch
import torch.nn.functional as F

from dquDataset import FashionIQ
from dquModel import DQU_CIR
from fiq_native_caption_math_031 import metrics, transitions
from run_fiq_native_caption_031 import identity, read, sha, write


CATEGORIES = {"dress": (2017, 3817), "shirt": (2038, 6346), "toptee": (1961, 5373)}
BATCH = 16
TOP_K = 100
CALIBRATION_QUERIES = 256
CALIBRATION_GALLERY = 2400
WEIGHTS = (0., .025, .05, .10, .20)
FIXED_WEIGHT = .05
LABEL = "DQU-CIR Flash 文本残差重排"
COMPARATIVE = re.compile(r"\b(?:lighter|darker|brighter|longer|shorter|larger|smaller|more|less)\b", re.I)


def sort_key(value):
    return hashlib.sha256(f"flash-residual-seed42:{value}".encode()).hexdigest()


def native_dataset(data, stage, category):
    dataset = object.__new__(FashionIQ)
    dataset.category = category
    dataset.image_dir = stage
    dataset.caption_dir = data / "captions"
    dataset.key_words = read(dataset.caption_dir / f"keywords_in_mods_{category}.json")
    dataset.correction_dict = read(dataset.caption_dir / f"correction_dict_{category}.json")
    return dataset


def stage_images(source, destination, names, report):
    destination.mkdir(exist_ok=True)
    paths = [(name, source._image_path(name)) for name in names]
    paths.sort(key=lambda item: (item[1].stat().st_dev, item[1].stat().st_ino))
    hashes = {}
    for i, (name, path) in enumerate(paths, 1):
        blob = path.read_bytes()
        hashes[name] = hashlib.sha256(blob).hexdigest()
        target = destination / f"{name}.jpg"
        if not target.exists() or sha(target) != hashes[name]:
            tmp = target.with_suffix(".copying")
            tmp.write_bytes(blob)
            tmp.replace(target)
        if i % 512 == 0 or i == len(paths):
            report(i, len(paths))
    return hashes


def load_rows(data, flash, inventory, category, split):
    raw = read(data / "captions" / f"cap.{category}.{split}.json")
    spec = inventory[f"{category}_{split}"]
    path = flash / "texts" / spec["path"]
    if sha(path) != spec["sha256"]:
        raise ValueError(f"Flash text export digest changed: {path}")
    with path.open(encoding="utf-8") as handle:
        exports = list(map(json.loads, handle))
    if len(raw) != spec["rows"] or len(exports) != len(raw):
        raise ValueError(f"Flash row count differs: {category}/{split}")
    rows = []
    for index, (item, export) in enumerate(zip(raw, exports)):
        if ((export["category"], export["split"], export["source_index"]) != (category, split, index)
                or export["fragments"] != item["captions"]
                or any(export[key] != item[key] for key in ("candidate", "target"))):
            raise ValueError(f"Flash query identity differs: {category}/{split}/{index}")
        rows.append(dict(item, source_index=index, flash=export["main_context_text"],
                         status=export["status"], reference_sha256=export["reference_sha256"]))
    return rows


def build_calibration(train, metadata, official_digests):
    hashes = metadata["image_sha256"]
    ids = set(metadata["banks"]["fit"] + metadata["banks"]["dev"])
    legal = {name for name in ids if hashes[name] not in official_digests}
    eligible = [row for row in train if {row["candidate"], row["target"]} <= legal
                and hashes[row["candidate"]] != hashes[row["target"]]]
    selected = sorted(eligible, key=lambda row: sort_key(row["source_index"]))[:CALIBRATION_QUERIES]
    if len(selected) != CALIBRATION_QUERIES:
        raise ValueError("Insufficient byte-isolated calibration queries")
    gallery = {row[key] for row in selected for key in ("candidate", "target")}
    seen = {hashes[name] for name in gallery}
    for name in sorted(legal - gallery, key=sort_key):
        if hashes[name] not in seen:
            gallery.add(name)
            seen.add(hashes[name])
        if len(gallery) == CALIBRATION_GALLERY:
            break
    if len(gallery) != CALIBRATION_GALLERY:
        raise ValueError("Insufficient byte-isolated calibration gallery")
    return selected, sorted(gallery), hashes


def encode(model, dataset, captions, rows, gallery, report):
    positions = {name: i for i, name in enumerate(gallery)}
    if len(positions) != len(gallery):
        raise ValueError("Duplicate gallery IDs")
    labels = torch.tensor([positions[row["target"]] for row in rows])
    refs = torch.tensor([positions[row["candidate"]] for row in rows])
    query, native, flash, targets = [], [], [], []

    def reference(row):
        return dataset.get_written_img(row["candidate"], row["target"], stage=1)[0]

    def target(name):
        return dataset.get_img(name, stage=1)[0]

    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            images = torch.stack(list(pool.map(reference, batch))).cuda()
            native_text = [captions[row["candidate"]] + ", but " + dataset._concat_text(row["captions"])
                           for row in batch]
            flash_text = [row["flash"] if row["status"] == "qwen_used" else text
                          for row, text in zip(batch, native_text)]
            with torch.autocast("cuda", dtype=torch.float16):
                query.append(model.extract_query(native_text, images).float().cpu())
                native.append(F.normalize(model.extract_text_fea(native_text), dim=-1).float().cpu())
                flash.append(F.normalize(model.extract_text_fea(flash_text), dim=-1).float().cpu())
            if start % 512 == 0 or start + len(batch) == len(rows):
                report("encode_query", start + len(batch), len(rows))
        for start in range(0, len(gallery), BATCH):
            batch = gallery[start:start + BATCH]
            images = torch.stack(list(pool.map(target, batch))).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                targets.append(model.extract_target(images).float().cpu())
            if start % 512 == 0 or start + len(batch) == len(gallery):
                report("encode_gallery", start + len(batch), len(gallery))
    tensors = [torch.cat(part) for part in (query, native, flash, targets)]
    if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise FloatingPointError("Nonfinite retrieval embedding")
    return dict(zip(("query", "native", "flash", "gallery"), tensors)), labels, refs


def prepare_rankings(features, labels, refs, active, top_k=TOP_K):
    base = features["query"] @ features["gallery"].T
    base[torch.arange(len(refs)), refs] = -torch.inf
    order = torch.stack([torch.argsort(row, descending=True) for row in base])
    original = (order == labels[:, None]).int().argmax(1) + 1
    candidates = order[:, :top_k]
    base_top = base.gather(1, candidates)
    delta = features["flash"] - features["native"]
    residual = torch.cat([
        torch.einsum("bd,bkd->bk", delta[i:i + 32], features["gallery"][candidates[i:i + 32]])
        for i in range(0, len(delta), 32)
    ])
    residual[~active] = 0
    return original, candidates, base_top, residual, labels


def ranks_at(prepared, weight):
    original, candidates, base, residual, labels = prepared
    if not weight:
        return original.tolist()
    within = torch.argsort(base + weight * residual, dim=1, descending=True, stable=True)
    reordered = candidates.gather(1, within)
    new = (reordered == labels[:, None]).int().argmax(1) + 1
    return torch.where(original <= candidates.shape[1], new, original).tolist()


def compare(rows, base, final):
    groups = {
        "all": list(range(len(rows))),
        "qwen_used": [i for i, row in enumerate(rows) if row["status"] == "qwen_used"],
        "fallback": [i for i, row in enumerate(rows) if row["status"] != "qwen_used"],
        "comparative_qwen": [i for i, row in enumerate(rows) if row["status"] == "qwen_used"
                             and COMPARATIVE.search(" ".join(row["captions"]))],
        "other_qwen": [i for i, row in enumerate(rows) if row["status"] == "qwen_used"
                       and not COMPARATIVE.search(" ".join(row["captions"]))],
    }
    result = {}
    for group, idx in groups.items():
        if idx:
            off, on = [base[i] for i in idx], [final[i] for i in idx]
            result[group] = dict(native=metrics(off), reranked=metrics(on),
                                 transitions={str(k): transitions(on, off, k) for k in (1, 10, 50)})
    return result


def save_rows(path, rows, baseline, selected, fixed):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("source_index", "candidate", "target", "text_status", "comparative",
                         "rank_native", "rank_selected", "rank_fixed_0.05"))
        for row, a, b, c in zip(rows, baseline, selected, fixed):
            writer.writerow((row["source_index"], row["candidate"], row["target"], row["status"],
                             bool(COMPARATIVE.search(" ".join(row["captions"]))), a, b, c))
    tmp.replace(path)


def store_features(path, features, labels, refs, rows):
    tmp = path.with_suffix(".tmp")
    torch.save(dict(features=features, labels=labels, refs=refs,
                    source_indices=[row["source_index"] for row in rows]), tmp)
    tmp.replace(path)


def active_rows(rows):
    return torch.tensor([row["status"] == "qwen_used" for row in rows], dtype=torch.bool)


def macro(results, method):
    return {f"r{k}": sum(result["all"][method][f"r{k}"] for result in results.values()) / len(results)
            for k in (1, 10, 50)}


def verify_gpu(gpu):
    import subprocess

    line = subprocess.check_output(
        ["nvidia-smi", "-i", "1", "--query-gpu=uuid,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True).strip()
    uuid, free, _ = [item.strip() for item in line.split(",")]
    if uuid != gpu or int(free) < 10240:
        raise RuntimeError(f"GPU 1 identity or free memory changed: {line}")


def run(root, cache, control, data, initial, gpu):
    state_path = root / "state.json"
    if state_path.exists():
        raise FileExistsError(state_path)
    state = dict(experiment=LABEL, status="running", gpu_index=1, gpu=gpu, identity=identity())

    def report(stage, category=None, done=0, total=0):
        state.update(stage=stage, category=category, done=done, total=total)
        write(state_path, state)
        if done == 0 or done == total:
            print(stage, category, done, total, flush=True)

    try:
        report("verify_inputs")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
            raise ValueError("GPU UUID differs from launch environment")
        verify_gpu(gpu)
        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        torch.cuda.set_per_process_memory_fraction(8192 * (1 << 20) / torch.cuda.get_device_properties(0).total_memory)
        flash = control / "flash"
        plan = read(flash / "plan.json")
        native_plan = read(control / "preflight.json")
        if read(control / "queue_state.json")["status"] != "complete":
            raise ValueError("Native checkpoint queue is incomplete")
        for name, digest in native_plan["native_code"].items():
            if sha(control / "dqu_code" / name) != digest:
                raise ValueError(f"Native DQU source digest changed: {name}")
        inventory = flash / "texts" / "inventory.json"
        if sha(inventory) != plan["text_inventory_sha256"]:
            raise ValueError("Flash inventory digest changed")
        inventory = read(inventory)["splits"]
        cache.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(cache).free < 20 * (1 << 30):
            raise RuntimeError("Less than 20 GiB free on NVMe")
        write(root / "plan.json", dict(experiment=LABEL, source=str(control), gpu=gpu,
                                      top_k=TOP_K, candidates=WEIGHTS, fixed_diagnostic=FIXED_WEIGHT,
                                      selection="calibration equal-category macro R10+R50, lower weight wins ties",
                                      calibration="original train, byte-disjoint from official gallery, "
                                                  "checkpoint previously trained on calibration queries",
                                      calibration_queries_per_category=CALIBRATION_QUERIES,
                                      calibration_gallery=CALIBRATION_GALLERY,
                                      score="native DQU score + weight * (Flash cosine - native text cosine)",
                                      seed=42, api_calls=0, script_sha256=sha(Path(__file__))))

        staged, calibration, digests = {}, {}, {}
        official_digests = set()
        for category, (count, gallery_count) in CATEGORIES.items():
            for path, expected in native_plan["native_sources"][category].items():
                if sha(Path(path)) != expected:
                    raise ValueError(f"Native data digest changed: {path}")
            stage = cache / category
            original = native_dataset(data, data / "resized_image", category)
            gallery = read(data / "image_splits" / f"split.{category}.val.json")
            if len(gallery) != gallery_count:
                raise ValueError(f"Official gallery count changed: {category}")
            report("stage_official_jpegs", category)
            val_hashes = stage_images(original, stage, gallery,
                                      lambda n, t: report("stage_official_jpegs", category, n, t))
            staged[category] = stage, gallery, val_hashes
            official_digests.update(val_hashes.values())
        for category, (stage, _, val_hashes) in staged.items():
            original = native_dataset(data, data / "resized_image", category)
            metadata = flash / "metadata" / f"{category}.json"
            if sha(metadata) != plan["inputs"][category]["metadata_sha256"]:
                raise ValueError(f"Training metadata changed: {category}")
            train = load_rows(data, flash, inventory, category, "train")
            rows, ids, expected = build_calibration(train, read(metadata), official_digests)
            report("stage_calibration_jpegs", category)
            cal_hashes = stage_images(original, stage, ids,
                                      lambda n, t: report("stage_calibration_jpegs", category, n, t))
            if any(digest != expected[name] for name, digest in cal_hashes.items()):
                raise ValueError(f"Calibration JPEG differs from original metadata: {category}")
            if set(cal_hashes.values()) & official_digests:
                raise ValueError(f"Calibration images overlap official gallery: {category}")
            if any(row["reference_sha256"] != cal_hashes[row["candidate"]] for row in rows):
                raise ValueError(f"Calibration Flash JPEG differs: {category}")
            calibration[category] = rows, ids
            digests[category] = dict(val=val_hashes, calibration=cal_hashes)
        write(root / "jpeg_sha256.json", digests)

        checkpoints, calibrations = {}, {}
        for category, (rows, gallery) in calibration.items():
            directory = root / category
            directory.mkdir()
            report("load_native_checkpoint_calibration", category)
            source, = (control / "dqu" / category / "checkpoints").glob("*best.pt")
            target = cache / source.name
            shutil.copyfile(source, target)
            if source.stat().st_size != target.stat().st_size:
                raise ValueError(f"Checkpoint copy size differs: {category}")
            checkpoints[category] = target
            model = load_model(initial, target)
            stage, _, _ = staged[category]
            dataset = native_dataset(data, stage, category)
            dataset.transform = (model.preprocess_val, model.preprocess_val)
            captions = read(dataset.caption_dir / f"image_captions_{category}_train.json")
            features, labels, refs = encode(model, dataset, captions, rows, gallery,
                                             lambda s, n, t: report("calibration_" + s, category, n, t))
            store_features(directory / "calibration_features.pt", features, labels, refs, rows)
            prepared = prepare_rankings(features, labels, refs, active_rows(rows))
            calibrations[category] = {str(w): metrics(ranks_at(prepared, w)) for w in WEIGHTS}
            write(directory / "calibration.json", dict(metrics=calibrations[category], gallery=gallery,
                                                        source_indices=[row["source_index"] for row in rows],
                                                        checkpoint_sha256=sha(target)))
            del model
            torch.cuda.empty_cache()
        scores = {w: sum(sum(calibrations[c][str(w)][f"r{k}"] for k in (10, 50))
                         for c in CATEGORIES) / len(CATEGORIES) for w in WEIGHTS}
        selected = max(WEIGHTS, key=lambda w: (scores[w], -w))
        write(root / "selection.json", dict(weight=selected, scores=scores, per_category=calibrations))

        results, fixed_results = {}, {}
        for category, (count, _) in CATEGORIES.items():
            stage, gallery, hashes = staged[category]
            rows = load_rows(data, flash, inventory, category, "val")
            if len(rows) != count or any(row["reference_sha256"] != hashes[row["candidate"]] for row in rows):
                raise ValueError(f"Official query identities differ: {category}")
            report("load_native_checkpoint_official", category)
            model = load_model(initial, checkpoints[category])
            dataset = native_dataset(data, stage, category)
            dataset.transform = (model.preprocess_val, model.preprocess_val)
            captions = read(dataset.caption_dir / f"image_captions_{category}_val.json")
            features, labels, refs = encode(model, dataset, captions, rows, gallery,
                                             lambda s, n, t: report("official_" + s, category, n, t))
            directory = root / category
            store_features(directory / "official_features.pt", features, labels, refs, rows)
            prepared = prepare_rankings(features, labels, refs, active_rows(rows))
            base, tuned, fixed = [ranks_at(prepared, w) for w in (0., selected, FIXED_WEIGHT)]
            old = read(next((control / "dqu" / category / "checkpoints").glob("*best.json")))
            if any(abs(metrics(base)[f"r{k}"] - old[f"{category}_r{k}"]) > 1e-8 for k in (1, 10, 50)):
                raise ValueError(f"Native recall reproduction differs: {category}")
            for i, row in enumerate(rows):
                if (row["status"] != "qwen_used" or base[i] > TOP_K) and (tuned[i] != base[i] or fixed[i] != base[i]):
                    raise ValueError("Reranking changed a query outside its allowed scope")
            results[category] = compare(rows, base, tuned)
            fixed_results[category] = compare(rows, base, fixed)
            save_rows(directory / "official_per_query.csv", rows, base, tuned, fixed)
            write(directory / "official_summary.json", dict(selected=results[category], fixed=fixed_results[category],
                                                             gallery=len(gallery), control_epoch=old["epoch"]))
            del model
            torch.cuda.empty_cache()
        write(root / "summary.json", dict(experiment=LABEL, selected_weight=selected, fixed_weight=FIXED_WEIGHT,
                                          native_macro=macro(results, "native"), selected_macro=macro(results, "reranked"),
                                          fixed_macro=macro(fixed_results, "reranked"), selected=results,
                                          fixed=fixed_results, api_calls=0))
        state.update(status="complete")
        report("complete")
    except BaseException as exc:
        state.update(status="failed", error=type(exc).__name__, detail=str(exc))
        write(state_path, state)
        raise


def load_model(initial, checkpoint):
    model = DQU_CIR(clip_checkpoint=str(initial))
    weights = torch.load(checkpoint, mmap=True, map_location="cpu", weights_only=True)
    model.load_state_dict(weights["model_state"], strict=True)
    return model.requires_grad_(False).cuda().eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "cache", "control", "data", "initial"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    run(args.root, args.cache, args.control, args.data, args.initial, args.gpu)
