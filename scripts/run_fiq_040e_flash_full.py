"""Run full Flash generation followed by the unchanged 040E learning protocol."""

import argparse
import fcntl
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from run_fiq_040e_flash import gpu_snapshot, read, sha
from run_fiq_context_generation_040e_full import COUNTS, launch, setup, write


MODEL = "qwen3.8-flash"
LABEL = "040E（Flash 全量）"
TRAIN = {"dress": 5639, "shirt": 5312, "toptee": 5291}
TRAINER = "run_fiq_vit_lora_039a_earlystop.py"
GPU_INDEX = 1


def prepare(root, data, paired):
    control = read(paired / "plan.json")
    if control["pilot"]["model"] != MODEL or control["arms"] != ["040E"]:
        raise ValueError("Expected the completed matched 040E Flash control")
    if read(paired / "experiment_state.json")["status"] != "complete":
        raise ValueError("Matched 040E Flash control is incomplete")
    for name, digest in control["source_sha256"].items():
        if name != TRAINER and sha(root / "code" / name) != digest:
            raise ValueError(f"Matched Flash training source differs: {name}")
    for category, spec in control["inputs"].items():
        meta_file = root / "metadata" / f"{category}.json"
        if sha(meta_file) != spec["metadata_sha256"] or len(read(meta_file)["records"]) != TRAIN[category]:
            raise ValueError(f"Full training metadata changed for {category}")
        for split in ("train", "val"):
            prefix = spec[f"{split}_prefix"]
            if Path(prefix["path"]).stat().st_size != prefix["bytes"]:
                raise ValueError(f"ViT prefix cache is missing or incomplete for {category}/{split}")
    if (root / "weights" / "open_clip_pytorch_model.bin").resolve(strict=True) != (
            paired / "weights" / "open_clip_pytorch_model.bin").resolve(strict=True):
        raise ValueError("Public OpenCLIP initialization differs")
    del control["pilot"]
    control.pop("caption_source", None)
    control["protocol"].update(
        training_queries=sum(TRAIN.values()),
        updates_per_epoch=sum(math.ceil(n / control["batch"]) for n in TRAIN.values()),
        validation=dict(queries=sum(v[1] for v in COUNTS.values()), split="full official FashionIQ val",
                        patience=3, criterion="strict equal-category macro R@10"),
    )
    control.update(experiment=LABEL, control="040E（Flash）", requested_gpu_index=GPU_INDEX,
                   control_plan_sha256=sha(paired / "plan.json"),
                   full_caption=dict(model=MODEL, generated_rows=sum(sum(n) for n in COUNTS.values()),
                                     train=TRAIN, validation={c: n[1] for c, n in COUNTS.items()}))
    control["source_sha256"] = {p.name: sha(p) for p in (root / "code").iterdir() if p.is_file()}
    write(root / "training_plan_template.json", control)
    setup(root, data, paired, workers=64, rpm=150, model=MODEL)


def train_after_generation(root):
    state_path = root / "experiment_state.json"
    generation = read(root / "state.json")
    if generation["status"] != "complete":
        write(state_path, dict(status="generation_incomplete", generation=generation))
        return
    texts = root / "texts" / "inventory.json"
    manifest = read(texts)
    plan = read(root / "training_plan_template.json")
    if manifest["rows"] != plan["full_caption"]["generated_rows"]:
        raise ValueError("Full text export count differs")
    for category, counts in COUNTS.items():
        for split, count in zip(("train", "val"), counts):
            spec = manifest["splits"][f"{category}_{split}"]
            if spec["rows"] != count or sha(root / "texts" / spec["path"]) != spec["sha256"]:
                raise ValueError(f"Full text export differs for {category}/{split}")
    first = gpu_snapshot()
    time.sleep(5)
    second = gpu_snapshot()
    gpu_index = plan["requested_gpu_index"]
    if gpu_index not in first or first[gpu_index] != second.get(gpu_index):
        write(state_path, dict(status="generation_complete_waiting_gpu", generation=generation,
                               requested_gpu_index=gpu_index, first=first, second=second))
        return
    gpu = second[gpu_index]
    training = root / "training"
    training.mkdir()
    for name in ("code", "texts", "metadata", "weights"):
        (training / name).symlink_to(root / name, target_is_directory=True)
    plan.update(gpu=gpu, gpu_index=gpu_index, text_inventory_sha256=sha(texts))
    plan["full_caption"]["generation_plan_sha256"] = sha(root / "plan.json")
    write(training / "plan.json", plan)
    write(state_path, dict(status="training", gpu_index=gpu_index, generation=generation))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
               PYTHONPATH=f"{root / 'code'}:/home/lcy/fashionCIR/src")
    with (training / "train.log").open("wb") as log:
        command = ["nice", "-n", "8", sys.executable, "-u", str(root / "code" / TRAINER),
                   "--root", str(training), "--gpu", gpu]
        result = subprocess.run(command, cwd="/home/lcy/fashionCIR", env=env, stdout=log, stderr=log)
    if result.returncode:
        write(state_path, dict(status="training_failed", returncode=result.returncode,
                               gpu_index=gpu_index, generation=generation))
        return
    summary = read(training / "summary.json")
    if read(training / "state.json")["status"] != "complete" or summary["last"]["stale"] != 3:
        raise ValueError("Full 040E did not complete normal early stopping")
    write(state_path, dict(status="complete", gpu_index=gpu_index, generation=generation, training=summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-from", type=Path)
    action.add_argument("--launch", action="store_true")
    action.add_argument("--resume-rejected", action="store_true")
    action.add_argument("--train-only", action="store_true")
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    if args.prepare_from:
        if args.data is None:
            parser.error("--prepare-from requires --data")
        prepare(args.root, args.data, args.prepare_from)
    elif args.train_only:
        with (args.root / "launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train_after_generation(args.root)
    else:
        launch(args.root, args.resume_rejected, after_complete=train_after_generation)
