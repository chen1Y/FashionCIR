"""DQU-CIR with the 040E image branch and native DQU text and batch loss."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import dquTest
from benchmark_data import selection_score
from dquDataset import FashionIQ
from dquModel import DQU_CIR
from dquTrain import set_seed, train_epoch
from experiment_storage import atomic_torch_save, require_space
from fiq_pixel_free_038 import VisualConflictEraserV2
from fiq_vit_lora_039 import attach_lora, tail_features
from run_029d_openclip_036 import OPENCLIP_SHA, gpu_snapshot, official_bank
from run_fiq_native_caption_031 import identity, read, sha, write


CATEGORIES = {"dress": (5985, 2017, 3817), "shirt": (5988, 2038, 6346), "toptee": (6027, 1961, 5373)}
SHAPE = (257, 1280)
BATCH = 16
SEED = 42
GPU_INDEX = 1
GPU_BUDGET_MIB = 18176
LABEL = "040 图像消融"


def stage_file(source, destination, expected):
    """Copy and verify large files in one sequential pass over the source disk."""
    if destination.exists() and destination.stat().st_size == source.stat().st_size and sha(destination) == expected:
        return
    tmp = destination.with_suffix(".copying")
    digest = hashlib.sha256()
    with source.open("rb") as src, tmp.open("wb") as dst:
        while block := src.read(16 * 1024 * 1024):
            digest.update(block)
            dst.write(block)
    if digest.hexdigest() != expected:
        tmp.unlink()
        raise ValueError(f"Source digest differs: {source}")
    tmp.replace(destination)


def original_rows(data, category, split):
    return read(data / "captions" / f"cap.{category}.{split}.json")


def drop_rows(flash, spec, raw, category, split):
    path = flash / "texts" / spec["path"]
    if sha(path) != spec["sha256"]:
        raise ValueError(f"Text export digest differs: {path}")
    with path.open(encoding="utf-8") as file:
        rows = list(map(json.loads, file))
    for index, (row, orig) in enumerate(zip(rows, raw, strict=True)):
        if ((row["category"], row["split"], row["source_index"]) != (category, split, index)
                or any(row[k] != orig[k] for k in ("candidate", "target"))
                or row["fragments"] != orig["captions"]):
            raise ValueError(f"Text identity differs for {category}/{split}/{index}")
    if len(rows) != spec["rows"]:
        raise ValueError(f"Text denominator differs: {path}")
    return rows


def image_paths(data, category):
    locator = object.__new__(FashionIQ)
    locator.image_dir = data / "resized_image"
    locator.category = category
    return locator


def add_missing(visual, transform, data, category, missing, expected):
    from PIL import Image
    import io

    locator = image_paths(data, category)
    parts = []

    def capture(_module, _inputs, output):
        hidden = output.transpose(0, 1).contiguous()
        if hidden.dtype != torch.bfloat16 or tuple(hidden.shape[1:]) != SHAPE:
            raise ValueError("Unexpected ViT prefix layout")
        parts.append(hidden.cpu().view(torch.uint16).numpy())

    hook = visual.transformer.resblocks[-3].register_forward_hook(capture)
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start in range(0, len(missing), 8):
                images = []
                for name in missing[start:start + 8]:
                    blob = locator._image_path(name).read_bytes()
                    if name in expected and hashlib.sha256(blob).hexdigest() != expected[name]:
                        raise ValueError(f"Reference image digest differs: {name}")
                    with Image.open(io.BytesIO(blob)) as image:
                        images.append(transform(image.convert("RGB")))
                visual(torch.stack(images).cuda())
    finally:
        hook.remove()
    return np.concatenate(parts) if parts else np.empty((0, *SHAPE), dtype=np.uint16)


class PrefixBank:
    def __init__(self, prefix, prefix_ids, extra, extra_ids):
        self.prefix = prefix
        self.extra = extra
        self.index = {name: i for i, name in enumerate([*prefix_ids, *extra_ids])}

    def batch(self, indices):
        count = len(self.prefix)
        values = np.empty((len(indices), *SHAPE), dtype=np.uint16)
        cached = indices < count
        if cached.any():
            values[cached] = self.prefix[indices[cached]]
        if (~cached).any():
            values[~cached] = self.extra[indices[~cached] - count]
        return torch.from_numpy(values).view(torch.bfloat16).cuda()


class QuerySet:
    """Expose the native training and validation fields using image indices."""

    def __init__(self, data, category, train, val, gallery, training_bank, validation_bank):
        native = object.__new__(FashionIQ)
        captions = data / "captions"
        native.correction_dict = read(captions / f"correction_dict_{category}.json")
        train_caps = read(captions / f"image_captions_{category}_train.json")
        val_caps = read(captions / f"image_captions_{category}_val.json")

        def text(caps, row):
            return caps[row["candidate"]] + ", but " + native._concat_text(row["captions"])

        train_targets = sorted({row["target"] for row in train})
        target_index = {name: i for i, name in enumerate(train_targets)}
        self.train_targets = train_targets
        self.train = [dict(visual_query=torch.tensor(i), textual_query=text(train_caps, row),
                           target_img_data=torch.tensor(target_index[row["target"]]))
                      for i, row in enumerate(train)]
        val_ids = {name: i for i, name in enumerate(gallery)}
        self.test_queries = [dict(visual_query=torch.tensor(i), textual_query=text(val_caps, row),
                                  source_img_id=val_ids[row["candidate"]], target_img_id=val_ids[row["target"]])
                             for i, row in enumerate(val)]
        self.test_targets = [dict(target_img_data=torch.tensor(i), target_img_id=i) for i in range(len(gallery))]
        self.train_refs = np.asarray([training_bank.index[row["candidate"]] for row in train])
        self.val_refs = np.asarray([validation_bank.index[row["candidate"]] for row in val])

    def __len__(self):
        return len(self.train)

    def __getitem__(self, index):
        return self.train[index]


class DQUImageAblation(DQU_CIR):
    def __init__(self, checkpoint):
        super().__init__(clip_checkpoint=str(checkpoint))
        self.clip.visual.requires_grad_(False)
        self.eraser = VisualConflictEraserV2()
        self.banks = {}

    def train(self, mode=True):
        super().train(mode)
        self.clip.visual.eval()
        return self

    def extract_query(self, textual_query, visual_query, *, return_scaler=False):
        bank, references, drop, active, _ = self.banks[self.training]
        rows = visual_query.cpu().numpy()
        text = F.normalize(self.extract_text_fea(textual_query), dim=-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reference = tail_features(self.clip.visual, bank.batch(references[rows]),
                                      checkpoint_blocks=self.training)
        image = self.eraser(reference, drop[visual_query], active[visual_query]).to(text.dtype)
        combined = self.combiner_fc(torch.cat((text, image), dim=-1))
        scaler = self.scaler_fc(self.dropout(combined))
        query = F.normalize(scaler * text + (1.0 - scaler) * image, dim=-1)
        return (query, scaler) if return_scaler else query

    def extract_target(self, target_img):
        return self.banks[self.training][4][target_img].to(torch.float16)


@torch.no_grad()
def frozen_targets(vision, bank, ids):
    indices = np.asarray([bank.index[name] for name in ids])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return torch.cat([tail_features(vision, bank.batch(indices[i:i + 8]))
                          for i in range(0, len(ids), 8)])


@torch.no_grad()
def frozen_drop(model, rows):
    phrases = sorted({row["drop_attrs"] for row in rows if row["drop_attrs"]})
    positions = {phrase: index + 1 for index, phrase in enumerate(phrases)}
    vectors = [torch.zeros((1, 1024), device="cuda")]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(phrases), 32):
            vectors.append(F.normalize(model.extract_text_fea(phrases[start:start + 32]).float(), dim=-1))
    indices = torch.tensor([positions.get(row["drop_attrs"], 0) for row in rows], device="cuda")
    return torch.cat(vectors)[indices], indices != 0


def stage_category(cache, plan, category, report):
    staged = {}
    for split in ("train", "val"):
        spec = plan["inputs"][category][f"{split}_prefix"]
        path = Path(spec["path"])
        if path.stat().st_size != spec["bytes"]:
            raise ValueError(f"Unexpected archived prefix size: {path}")
        destination = cache / f"{category}_{split}.npy"
        report("stage_prefix_nvme", category=category, split=split, bytes=spec["bytes"])
        stage_file(path, destination, spec["sha256"])
        staged[split] = destination
    return staged


def setup_category(model, category, data, flash, plan, inventory, staged, report):
    train = original_rows(data, category, "train")
    val = original_rows(data, category, "val")
    gallery = read(data / "image_splits" / f"split.{category}.val.json")
    if (len(train), len(val), len(gallery)) != CATEGORIES[category]:
        raise ValueError(f"Original DQU denominator differs: {category}")
    metadata = flash / "metadata" / f"{category}.json"
    if sha(metadata) != plan["inputs"][category]["metadata_sha256"]:
        raise ValueError(f"Metadata digest changed: {category}")
    meta = read(metadata)
    train_ids = list(dict.fromkeys(meta["banks"]["fit"] + meta["banks"]["dev"]))
    archived_val, val_hashes, _ = official_bank(category)
    if set(archived_val["ids"]) != set(gallery):
        raise ValueError(f"Official gallery differs: {category}")
    archives = {}
    counts = {}
    for split, raw, ids, hashes in (
        ("train", train, train_ids, meta["image_sha256"]),
        ("val", val, archived_val["ids"], val_hashes),
    ):
        rows = drop_rows(flash, inventory[f"{category}_{split}"], raw, category, split)
        prefix = np.load(staged[split], mmap_mode="r", allow_pickle=False)
        if prefix.dtype != np.uint16 or prefix.shape != (len(ids), *SHAPE):
            raise ValueError(f"Cache shape changed: {category}/{split}")
        needed = {row[k] for row in rows for k in ("candidate", "target")}
        missing = sorted(needed - set(ids))
        known_ref_sha = {}
        for row in rows:
            name = row["candidate"]
            digest = row["reference_sha256"]
            if (name in hashes and hashes[name] != digest) or known_ref_sha.setdefault(name, digest) != digest:
                raise ValueError(f"Reference digest changed: {name}")
        report("supplement_original_split", category=category, split=split, missing=len(missing))
        extra = add_missing(model.clip.visual, model.preprocess_val, data, category, missing,
                            {**known_ref_sha, **hashes})
        drop, active = frozen_drop(model, rows)
        archives[split] = PrefixBank(prefix, ids, extra, missing), drop, active
        counts[split] = dict(queries=len(rows), missing=len(missing), drop_active=int(active.sum()),
                             text_status=dict(Counter(row["status"] for row in rows)))

    train_bank, train_drop, train_active = archives["train"]
    val_bank, val_drop, val_active = archives["val"]
    dataset = QuerySet(data, category, train, val, gallery, train_bank, val_bank)
    report("encode_frozen_targets", category=category)
    train_targets = frozen_targets(model.clip.visual, train_bank, dataset.train_targets)
    val_targets = frozen_targets(model.clip.visual, val_bank, gallery)
    model.banks = {True: (train_bank, dataset.train_refs, train_drop, train_active, train_targets),
                   False: (val_bank, dataset.val_refs, val_drop, val_active, val_targets)}
    return dataset, counts


def save_rows(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def fit(model, dataset, category, directory, report, adapters):
    lora = [parameter for adapter in adapters for parameter in adapter.parameters()]
    native_text = [p for n, p in model.clip.named_parameters() if p.requires_grad and not n.startswith("visual.")]
    head = [p for n, p in model.named_parameters() if not n.startswith(("clip.", "eraser."))]
    optimizer = torch.optim.AdamW([dict(params=native_text, lr=1e-6), dict(params=head, lr=1e-4),
                                   dict(params=lora, lr=1e-4), dict(params=model.eraser.parameters(), lr=1e-4)],
                                  weight_decay=.01, foreach=False)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=4,
                        generator=generator, pin_memory=True)
    scaler = torch.cuda.amp.GradScaler()
    params = SimpleNamespace(batch_size=BATCH, smoke_train_batches=0)
    rows, best, stale = [], None, 0
    for epoch in range(1, 101):
        if epoch == 8:
            for group in optimizer.param_groups:
                group["lr"] *= .1
        report("train", category=category, epoch=epoch)
        loss = train_epoch(params, model, optimizer, loader, torch.device("cuda"), scaler, epoch)
        if not np.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss: {category}/{epoch}")
        report("official_validation", category=category, epoch=epoch)
        recalls = dict(dquTest.test(params, model, dataset, category))
        score = selection_score(recalls, category)
        improved = best is None or score > best["selection_score"]
        row = dict(epoch=epoch, category=category, train_loss=loss,
                   queries=len(dataset.test_queries), gallery=len(dataset.test_targets),
                   **{k.removeprefix(f"{category}_"): v for k, v in recalls.items()},
                   selection_score=score, gamma=float(model.eraser.gamma.detach().clamp(0., 1.5)), best=improved)
        rows.append(row)
        save_rows(directory / "validation_metrics.csv", rows)
        if improved:
            best, stale = row, 0
            atomic_torch_save(dict(epoch=epoch, metrics=row,
                                   trainable={n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}),
                              directory / "best_delta.pt")
            write(directory / "best.json", best)
        else:
            stale += 1
        if stale >= 3:
            break
    return dict(best=best, epochs=len(rows), stale=stale)


def run(root, cache, control, data, gpu):
    state_file = root / "state.json"
    if state_file.exists():
        raise FileExistsError(state_file)
    state = dict(experiment=LABEL, status="running", identity=identity(), gpu_index=GPU_INDEX,
                 gpu=gpu, completed={})

    def report(stage, **details):
        for key in ("category", "split", "bytes", "missing", "epoch"):
            state.pop(key, None)
        state.update(stage=stage, **details)
        write(state_file, state)
        print(stage, details, flush=True)

    try:
        report("verify_protocol")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
            raise ValueError("GPU UUID differs from the launch environment")
        flash = control / "flash"
        plan = read(flash / "plan.json")
        baseline = read(control / "preflight.json")
        if read(control / "queue_state.json")["status"] != "complete" or plan["arms"] != ["040E"]:
            raise ValueError("Expected the completed 040H control")
        for name, digest in baseline["native_code"].items():
            if sha(control / "dqu_code" / name) != digest:
                raise ValueError(f"Native DQU source digest changed: {name}")
        for name, digest in plan["source_sha256"].items():
            if sha(flash / "code" / name) != digest:
                raise ValueError(f"040E source digest changed: {name}")
        manifest = flash / "texts" / "inventory.json"
        if sha(manifest) != plan["text_inventory_sha256"]:
            raise ValueError("Flash inventory digest changed")
        inventory = read(manifest)["splits"]
        control_bests = {}
        for category in CATEGORIES:
            for path, digest in baseline["native_sources"][category].items():
                if sha(Path(path)) != digest:
                    raise ValueError(f"Native data digest changed: {path}")
            best_path, = (control / "dqu" / category / "checkpoints").glob("*best.json")
            control_bests[category] = read(best_path)
        write(root / "plan.json", dict(experiment=LABEL, control=str(control), clip_sha256=OPENCLIP_SHA,
                                      flash_plan_sha256=sha(flash / "plan.json"), texts_sha256=sha(manifest),
                                      script_sha256=sha(Path(__file__)), seed=SEED, batch=BATCH, max_epochs=100,
                                      patience=3, native_text_lr=1e-6, native_fusion_lr=1e-4, lora_lr=1e-4,
                                      eraser_lr=1e-4, visual_precision="BF16", api_calls=0, control_best=control_bests,
                                      nvme_cache=str(cache), gpu=gpu, gpu_budget_mib=GPU_BUDGET_MIB))
        require_space(root, 20)
        cache.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(cache).free < 50 * (1 << 30):
            raise RuntimeError("NVMe has less than 50 GiB free for the staged inputs")
        checkpoint = cache / "open_clip_pytorch_model.bin"
        report("stage_public_checkpoint_nvme")
        stage_file(control / "weights" / checkpoint.name, checkpoint, OPENCLIP_SHA)
        torch.set_num_threads(4)
        for category in CATEGORIES:
            directory = root / category
            directory.mkdir()
            staged = stage_category(cache, plan, category, report)
            report("check_gpu", category=category)
            card, _ = gpu_snapshot(gpu, GPU_INDEX)
            if card["free_mib"] < GPU_BUDGET_MIB + 1024:
                raise RuntimeError(f"GPU 0 has insufficient free memory: {card['free_mib']} MiB")
            torch.cuda.set_per_process_memory_fraction(GPU_BUDGET_MIB * (1 << 20)
                                                       / torch.cuda.get_device_properties(0).total_memory)
            set_seed(SEED)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            report("load_model", category=category)
            model = DQUImageAblation(checkpoint).cuda().eval()
            model.clip.transformer.grad_checkpointing = True
            dataset, counts = setup_category(model, category, data, flash, plan, inventory, staged, report)
            write(directory / "inputs.json", counts)
            set_seed(SEED)
            adapters = list(attach_lora(model.clip.visual))
            result = fit(model, dataset, category, directory, report, adapters)
            control_values = control_bests[category]
            result["delta_vs_native_pp"] = {f"r{k}": result["best"][f"r{k}"] - control_values[f"{category}_r{k}"]
                                            for k in (1, 10, 50)}
            write(directory / "summary.json", result)
            state["completed"][category] = result
            del adapters, model, dataset
            torch.cuda.empty_cache()
        write(root / "summary.json", dict(experiment=LABEL, results=state["completed"], api_calls=0))
        state.update(status="complete", stage="complete")
        write(state_file, state)
    except BaseException as exc:
        state.update(status="failed", error=type(exc).__name__, detail=str(exc))
        write(state_file, state)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "cache", "control", "data"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    run(args.root, args.cache, args.control, args.data, args.gpu)
