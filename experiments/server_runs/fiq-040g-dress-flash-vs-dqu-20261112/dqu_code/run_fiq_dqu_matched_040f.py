"""Train native DQU-CIR on the exact 040C/D/E pilot queries and galleries."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import torch
from torch.utils.data import ConcatDataset, DataLoader

from dquDataset import FashionIQ
from dquModel import DQU_CIR
from dquTrain import set_seed
from experiment_storage import atomic_torch_save, require_space


CATEGORIES = {"dress": 3817, "shirt": 6346, "toptee": 5373}
SEED = 42
BATCH = 16
PATIENCE = 3
KS = (1, 10, 50)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def verify_inputs(control, data):
    plan = read(control / "plan.json")
    cases_file = control / "cases.json"
    if sha(cases_file) != plan["pilot"]["cases_sha256"]:
        raise ValueError("040C/D/E cases hash mismatch")
    cases = read(cases_file)
    expected = plan.get("comparison_queries", {cat: {"train": 128, "val": 64} for cat in CATEGORIES})
    offsets, sources, galleries = {}, {}, {}
    for category, counts in expected.items():
        gallery_size = CATEGORIES[category]
        metadata_file = control / "metadata" / f"{category}.json"
        if sha(metadata_file) != plan["inputs"][category]["metadata_sha256"]:
            raise ValueError(f"040C/D/E training metadata hash mismatch for {category}")
        metadata = read(metadata_file)
        gallery_file = data / "image_splits" / f"split.{category}.val.json"
        gallery = read(gallery_file)
        if len(gallery) != gallery_size or len(set(gallery)) != gallery_size:
            raise ValueError(f"Unexpected official gallery for {category}")
        galleries[category] = dict(count=gallery_size, sha256=sha(gallery_file))
        offsets[category] = {}
        keywords = read(data / "captions" / f"keywords_in_mods_{category}.json")
        for split in ("train", "val"):
            count = counts[split]
            selected = [r for r in cases if (r["category"], r["split"]) == (category, split)]
            cap_file = data / "captions" / f"cap.{category}.{split}.json"
            captions = read(cap_file)
            ref_captions = read(data / "captions" / f"image_captions_{category}_{split}.json")
            if len(selected) != count or [r["offset"] for r in selected] != plan["pilot"]["offsets"][category][split]:
                raise ValueError(f"040C/D/E offset mismatch for {category}/{split}")
            indices = [r["source_index"] for r in selected]
            if len(set(indices)) != count:
                raise ValueError(f"Duplicate query for {category}/{split}")
            for case, index in zip(selected, indices):
                item = captions[index]
                ref, target = item["candidate"], item["target"]
                if ref != case["candidate"] or item["captions"] != case["fragments"]:
                    raise ValueError(f"Original annotation mismatch for {category}/{split}/{index}")
                if split == "train":
                    prior = metadata["records"][case["offset"]]
                    if (prior["source_index"], prior["candidate"], prior["target"]) != (index, ref, target):
                        raise ValueError(f"Training pair mismatch for {category}/{index}")
                elif case["offset"] != index or ref not in gallery or target not in gallery:
                    raise ValueError(f"Validation pair mismatch for {category}/{index}")
                ref_path = data / "resized_image" / f"{ref}.jpg"
                target_path = data / "resized_image" / f"{target}.jpg"
                if sha(ref_path) != case["reference_sha256"] or not target_path.is_file():
                    raise ValueError(f"Reference or target file mismatch for {category}/{split}/{index}")
                if ref not in ref_captions or f"{ref}_{target}" not in keywords:
                    raise ValueError(f"Missing native DQU input for {category}/{split}/{index}")
            offsets[category][split] = indices
            sources[f"{category}_{split}"] = sha(cap_file)
    return dict(cases_sha256=sha(cases_file), control_plan_sha256=sha(control / "plan.json"),
                sources=sources, offsets=offsets, galleries=galleries)


class MatchedFashionIQ(FashionIQ):
    """Filter only the original triplet lists, preserving native DQU transforms."""

    def __init__(self, data, category, transform, indices):
        self.indices = {f"cap.{category}.{split}.json": rows for split, rows in indices.items()}
        super().__init__(str(data), category, transform, split="original-split")

    def _load_json(self, path):
        rows = super()._load_json(path)
        indices = self.indices.get(path.name)
        return [rows[i] for i in indices] if indices is not None else rows


class CategoryBatches:
    def __init__(self, counts):
        self.counts = list(counts)
        self.rng = torch.Generator().manual_seed(SEED)

    def __len__(self):
        return sum(math.ceil(n / BATCH) for n in self.counts)

    def __iter__(self):
        batches = []
        offset = 0
        for count in self.counts:
            order = torch.randperm(count, generator=self.rng) + offset
            batches.extend(order.split(BATCH))
            offset += count
        for index in torch.randperm(len(batches), generator=self.rng):
            yield batches[index].tolist()


def train_epoch(model, optimizer, scaler, loader, report):
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
    losses, skipped = [], 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model.compute_loss(batch["textual_query"], batch["visual_query"].cuda(),
                                      batch["target_img_data"].cuda())["ranking"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite native DQU ranking loss")
        scaler.scale(loss).backward()
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped += scaler.get_scale() < scale
        losses.append(loss.item())
        report("train", batches=len(losses), skipped_updates=skipped, loss=losses[-1])
    optimizer.zero_grad(set_to_none=True)
    if len(losses) != len(loader) or skipped == len(losses):
        raise RuntimeError("Native DQU did not train on the complete matched epoch")
    return sum(losses) / len(losses), skipped


@torch.inference_mode()
def evaluate(model, dataset, gallery_batch, report):
    model.eval()
    queries, targets = dataset.test_queries, dataset.test_targets
    query_features, target_features = [], []
    with torch.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(queries), BATCH):
            batch = queries[start:start + BATCH]
            visual = torch.stack([item["visual_query"] for item in batch]).cuda()
            query_features.append(model.extract_query([item["textual_query"] for item in batch], visual).float().cpu())
        for start in range(0, len(targets), gallery_batch):
            batch = targets[start:start + gallery_batch]
            images = torch.stack([item["target_img_data"] for item in batch]).cuda()
            target_features.append(model.extract_target(images).float().cpu())
            if start % (gallery_batch * 16) == 0:
                report("gallery", gallery_done=start + len(batch), gallery_total=len(targets))
    scores = torch.cat(query_features) @ torch.cat(target_features).T
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Nonfinite retrieval scores")
    ref = [item["source_img_id"] for item in queries]
    target = [item["target_img_id"] for item in queries]
    row = torch.arange(len(queries))
    scores[row, ref] = -torch.inf
    order = scores.argsort(dim=1, descending=True)
    return (order == torch.tensor(target)[:, None]).to(torch.int32).argmax(dim=1).add(1).tolist()


def metrics(ranks):
    return {f"r{k}": 100 * sum(rank <= k for rank in ranks) / len(ranks) for k in KS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "control", "data", "checkpoint"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    root = args.root
    inventory = verify_inputs(args.control, args.data)
    if args.preflight:
        print(json.dumps(inventory))
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu or torch.cuda.device_count() != 1:
        raise RuntimeError("Bind the run to one verified GPU UUID")
    if (root / "state.json").exists():
        raise FileExistsError("Run directory already contains an experiment")
    state = dict(status="running", pid=os.getpid(), gpu=args.gpu)

    def report(stage, **details):
        state.update(stage=stage, **details)
        write(root / "state.json", state)

    try:
        report("initialize")
        require_space(root, 16)
        write(root / "plan.json", dict(
            experiment=read(args.control / "plan.json").get("experiment", "040F matched native DQU-CIR"),
            **inventory,
            seed=SEED, batch=BATCH, val_batch=64, patience=PATIENCE, api_calls=0,
            pretrained=str(args.checkpoint.resolve()), pretrained_bytes=args.checkpoint.stat().st_size,
            backbone="ViT-H-14", clip_lr=1e-6, fusion_lr=1e-4, weight_decay=.01,
            loss="native one-way in-batch NCE", selection="equal-category val macro R@10",
            shared_model=True, trainable="all CLIP and native fusion", precision="FP16 AMP",
            source_sha256={p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
        ))
        set_seed(SEED)
        cv2.setNumThreads(1)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = DQU_CIR(clip_checkpoint=str(args.checkpoint)).cuda()
        model.clip.set_grad_checkpointing(True)
        clip_params = list(model.clip.parameters())
        fusion_params = [p for n, p in model.named_parameters() if not n.startswith("clip.")]
        optimizer = torch.optim.AdamW([dict(params=clip_params, lr=1e-6),
                                       dict(params=fusion_params, lr=1e-4)],
                                      weight_decay=.01, foreach=False)
        scaler = torch.amp.GradScaler("cuda")
        report("prepare_inputs", trainable_parameters=sum(p.numel() for p in model.parameters()))
        transform = (model.preprocess_train, model.preprocess_val)
        datasets = {}
        for category, gallery_spec in inventory["galleries"].items():
            expected_gallery = gallery_spec["count"]
            report("prepare_inputs", category=category)
            dataset = MatchedFashionIQ(args.data, category, transform, inventory["offsets"][category])
            indices = inventory["offsets"][category]
            if (len(dataset), len(dataset.test_queries), len(dataset.test_targets)) != (
                len(indices["train"]), len(indices["val"]), expected_gallery
            ):
                raise ValueError(f"DQU dataset counts changed for {category}")
            if [item["target_img_id"] for item in dataset.test_targets] != list(range(expected_gallery)):
                raise ValueError(f"Native gallery ID order changed for {category}")
            datasets[category] = dataset
        loader = DataLoader(ConcatDataset(list(datasets.values())),
                            batch_sampler=CategoryBatches(map(len, datasets.values())),
                            num_workers=0, generator=torch.Generator().manual_seed(SEED))
        best, stale, epoch, updates = None, 0, 0, 0
        history = []
        while stale < PATIENCE:
            epoch += 1
            report("train", epoch=epoch, batches=0)
            if epoch == 8:
                for group in optimizer.param_groups:
                    group["lr"] *= .1
            loss, skipped = train_epoch(model, optimizer, scaler, loader, report)
            updates += len(loader) - skipped
            rows = {}
            for category, dataset in datasets.items():
                report("validate", category=category)
                ranks = evaluate(model, dataset, 64, report)
                rows[category] = dict(**metrics(ranks), ranks=ranks, gallery=len(dataset.test_targets))
            all_ranks = [rank for row in rows.values() for rank in row["ranks"]]
            score = sum(rank <= 10 for rank in all_ranks)
            result = dict(epoch=epoch, updates=updates, skipped_updates=skipped, loss=loss,
                          macro=metrics(all_ranks), rows=rows)
            write(root / f"epoch_{epoch:04d}.json", result)
            if best is None or score > best["hits_at_10"]:
                stale = 0
                best = dict(epoch=epoch, hits_at_10=score, macro=result["macro"], updates=updates)
                report("save_best", best_epoch=epoch)
                require_space(root, 16)
                atomic_torch_save(dict(epoch=epoch, model_state=model.state_dict(),
                                       metrics=result["macro"]), root / "best.pt")
            else:
                stale += 1
            history.append(dict(epoch=epoch, updates=updates, loss=loss,
                                skipped_updates=skipped, macro=result["macro"], stale=stale))
            write(root / "history.json", history)
            write(root / "summary.json", dict(best=best, last=history[-1]))
            print(json.dumps(history[-1]), flush=True)
        state["status"] = "complete"
        report("complete", epoch=epoch, updates=updates, best_epoch=best["epoch"])
    except BaseException as exc:
        state["status"] = "failed"
        report(state["stage"], error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
