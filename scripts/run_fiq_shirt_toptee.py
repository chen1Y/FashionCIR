"""Original DQU and three v7 branches; online Dress-protocol texts, no additions."""
from __future__ import annotations

import argparse
import datetime
import getpass
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experiment_storage import require_space
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json, load_fashioniq_queries
from generate_fashioniq_online import CATEGORIES, EXPECTED, check_complete
from run_cirr_shoes_benchmark import detach_queue


def publish_checkpoint(source, destination):
    """One final, bit-identical persistent copy; no precision/storage format change."""
    if destination.exists() or destination.with_suffix(".json").exists():
        raise FileExistsError("Inspect existing persistent checkpoint; never overwrite")
    require_space(destination.parent, 8 + source.stat().st_size / 2**30 + .05)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise FileExistsError("Inspect interrupted checkpoint publication")
    with source.open("rb") as src, temporary.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=4 * 2**20)
        dst.flush()
        os.fsync(dst.fileno())
    checksum = file_sha(source)
    if file_sha(temporary) != checksum:
        raise ValueError("Persistent checkpoint SHA mismatch; staging original retained")
    os.replace(temporary, destination)
    shutil.copyfile(source.with_suffix(".json"), destination.with_suffix(".json"))
    return checksum


def plan(root, output, clip, staging):
    commands = {}
    for category in CATEGORIES:
        folder = output / category
        checkpoints = folder / "checkpoints"
        base = checkpoints / f"{category}_dqu_original_seed42_best.pt"
        common = ["--dataset", category, "--fashioniq-path", str(root), "--fashioniq-split", "original-split",
                  "--clip-checkpoint", str(clip), "--seed", "42", "--batch-size", "16", "--num-workers", "4"]
        commands[category] = [("dqu_original", [sys.executable, "src/dquTrain.py", *common,
            "--model-dir", str(staging / category), "--run-name", "dqu_original", "--num-epochs", "100",
            "--patience", "5", "--lr", "1e-4", "--clip-lr", "1e-6"])]
        for mode in ("off", "identity", "annotated"):
            commands[category].append(("structured_" + mode, [sys.executable, "src/newTrain.py", *common,
                "--model-dir", str(checkpoints), "--dqu-checkpoint", str(base), "--freeze-dqu-backbone",
                "--structured-train-path", str(folder / f"structured_{category}_train.json"),
                "--structured-val-path", str(folder / f"structured_{category}_val.json"),
                "--run-name", "structured_" + mode, "--multi-positive", mode,
                "--hard-negative-weight", "0", "--num-epochs", "6", "--patience", "7"]))
    return commands


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--clip-checkpoint", type=Path, required=True)
    p.add_argument("--staging", type=Path, default=Path("/dev/shm/fiq_shirt_toptee_20261022"))
    p.add_argument("--detach", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.root, args.output, args.clip_checkpoint = args.root.resolve(), args.output.resolve(), args.clip_checkpoint.resolve()
    if not args.staging.is_absolute() or args.staging.parent != Path("/dev/shm"):
        raise ValueError("Checkpoint staging must be a dedicated direct child of /dev/shm")
    commands = plan(args.root, args.output, args.clip_checkpoint, args.staging)
    source_hashes = {}
    audit = {}
    for category in CATEGORIES:
        audit[category] = {}
        for split in ("train", "val", "test"):
            caption = args.root / "captions" / f"cap.{category}.{split}.json"
            source_hashes[str(caption)] = file_sha(caption)
            split_path = args.root / "image_splits" / f"split.{category}.{split}.json"
            source_hashes[str(split_path)] = file_sha(split_path)
            if split == "test":
                continue
            rows = load_fashioniq_queries(args.root, category, split)
            assert len(rows) == EXPECTED[category][split]
            ids = set(json.loads(split_path.read_text()))
            assert all(r["candidate"] in ids and r["target"] in ids for r in rows)
            missing = [r["reference_image"] for r in rows if not Path(r["reference_image"]).is_file()]
            assert not missing, missing[:3]
            audit[category][split] = len(rows)
        gallery = json.loads((args.root / "image_splits" / f"split.{category}.val.json").read_text())
        audit[category]["gallery"] = len(gallery)
        # Includes target images and complete val gallery, not only API references.
        all_ids = set(gallery) | set(json.loads((args.root / "image_splits" / f"split.{category}.train.json").read_text()))
        assert all((args.root / "resized_image" / f"{i}.jpg").is_file() for i in all_ids)
        for name in (f"correction_dict_{category}.json", f"image_captions_{category}_train.json",
                     f"image_captions_{category}_val.json", f"keywords_in_mods_{category}.json"):
            path = args.root / "captions" / name
            source_hashes[str(path)] = file_sha(path)
    protocol = dict(categories=list(CATEGORIES), dataset_audit=audit, data_expansion=False, seed=42,
        baseline=dict(maximum_epochs=100, early_stopping_patience=5, clip_lr=1e-6, fusion_lr=1e-4, amp="fp16"),
        structured=dict(epochs=6, gate_epochs=2, adapter_epochs=4, modes=["off", "identity", "annotated"],
                        hard_negative_weight=0, frozen_category_dqu=True, lr=5e-5, train_amp="bf16", eval_amp="fp16"),
        selection="R@10+R@50 on unchanged full original-split validation", batch_size=16,
        online=dict(model="qwen3.7-flash", max_total_cny=15, max_category_cny=7.5,
                    max_total_calls=32028, max_tokens=1024, workers=8, rpm=120,
                    prompt="Original Dress SYSTEM_PROMPT/build_user_prompt and deterministic rules",
                    query_inputs="Reference image and original modification only; never target image",
                    maximum_known_invalid_repairs=1, network_retries=0, minimum_valid_coverage=.9),
        maximum_wall_seconds=43200, checkpoint_storage="Unmodified torch FP32 state; tmpfs atomic bests, SHA-verified final durable copy",
        source_hashes=source_hashes, clip_sha256=file_sha(args.clip_checkpoint),
        code_hashes={str(f.relative_to(ROOT)):file_sha(f) for f in [*sorted((ROOT / "src").glob("*.py")),
                     ROOT / "scripts/run_fiq_shirt_toptee.py", ROOT / "scripts/run_cirr_shoes_benchmark.py"]
                     if f.name not in ("generateModify.py", "generatetext.py", "generatetextbyimg.py")},
        commands=commands, final_scope="Per-category single-seed DQU/off/identity/annotated; no significance claim")
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "experiment_report.json").exists() or args.staging.exists():
        raise FileExistsError("Inspect previous queue/staging; never launch twice")
    require_space(args.output, 8 + 2 * 3.7 + .5)
    require_space(Path("/dev/shm"), 16)
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("GPU occupied; queue not launched")
    import fcntl
    lock = (args.output / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    key = getpass.getpass("DashScope API key (memory only): ")
    if not key:
        raise ValueError("Empty API key")
    if args.detach and not detach_queue(args.output):
        return
    args.staging.mkdir()
    (args.output / "queue.pid").write_text(str(os.getpid()) + "\n")
    atomic_write_json(args.output / "protocol.json", protocol)
    report = dict(status="starting", started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  queue_pid=os.getpid(), stages=[], dataset_audit=audit, metrics={}, costs={}, checkpoint_hashes={})
    started = time.monotonic()
    children = []
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")

    def save(**values):
        report.update(values)
        report["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cost = args.output / "cost_summary.json"
        if cost.exists():
            report["costs"] = json.loads(cost.read_text())
        atomic_write_json(args.output / "experiment_report.json", report)

    def guard():
        require_space(args.output)
        if time.monotonic() - started > 43200:
            raise TimeoutError("12-hour queue wall budget reached; no automatic restart")
        if generator.poll() not in (None, 0):
            raise RuntimeError("Online generation failed; see online_generation.log and retained ledger")

    def stop_child(child):
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()

    read_fd, write_fd = os.pipe()
    try:
        with (args.output / "online_generation.log").open("x") as log:
            generator = subprocess.Popen([sys.executable, "src/generate_fashioniq_online.py", "--root", str(args.root),
                "--output", str(args.output), "--api-key-fd", str(read_fd)], cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, pass_fds=(read_fd,), start_new_session=True)
        children.append(generator)
        os.close(read_fd)
        os.write(write_fd, key.encode())
        os.close(write_fd)
        key = ""
        save(generation_pid=generator.pid)
        for category in CATEGORIES:
            folder = args.output / category
            checkpoint_dir = folder / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            report["metrics"][category] = {}
            for name, command in commands[category]:
                guard()
                if name.startswith("structured_"):
                    save(status=f"waiting_online_{category}")
                    while True:
                        ready = all((folder / f"structured_{category}_{s}.json").exists()
                                    and json.loads((folder / f"structured_{category}_{s}.json").read_text()).get("state") == "complete"
                                    for s in ("train", "val"))
                        if ready:
                            break
                        guard()
                        if generator.poll() == 0:
                            raise ValueError("Online process exited without complete category records")
                        time.sleep(10)
                    for split in ("train", "val"):
                        check_complete(json.loads((folder / f"structured_{category}_{split}.json").read_text()),
                                       load_fashioniq_queries(args.root, category, split), category, split)
                row = dict(category=category, stage=name, status="running", command=command,
                           started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                report["stages"].append(row)
                log_path = folder / (name + ".log")
                with log_path.open("x") as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    children.append(child)
                    row["pid"] = child.pid
                    save(status=f"{category}_{name}")
                    while child.poll() is None:
                        guard()
                        with log_path.open("rb") as stream:
                            stream.seek(max(0, log_path.stat().st_size - 10000))
                            tail = stream.read().decode(errors="replace")
                        if re.search(r"(?:loss|train_loss)=(?:nan|inf)\b|CUDA out of memory|OutOfMemoryError", tail, re.I):
                            raise RuntimeError("Non-finite loss/OOM detected; preserve partial stage and stop")
                        time.sleep(10)
                row.update(returncode=child.returncode, status="complete" if child.returncode == 0 else "failed")
                save()
                if child.returncode:
                    raise RuntimeError(f"{category}/{name} failed; no next stage")
                if name == "dqu_original":
                    source = args.staging / category / f"{category}_dqu_original_seed42_best.pt"
                    destination = checkpoint_dir / source.name
                    report["checkpoint_hashes"][category] = publish_checkpoint(source, destination)
                    base_json = source.with_suffix(".json")
                    # Staging retains no authoritative copy after verified durable publication.
                    source.unlink()
                    base_json.unlink()
                    metric = json.loads(destination.with_suffix(".json").read_text())
                else:
                    metric = json.loads((checkpoint_dir / f"{category}_{name}_seed42_best.json").read_text())
                for k in (f"{category}_r1", f"{category}_r10", f"{category}_r50"):
                    if not 0 <= metric[k] <= 100:
                        raise ValueError("Invalid final metrics")
                report["metrics"][category][name] = metric
                save()
            base = checkpoint_dir / f"{category}_dqu_original_seed42_best.pt"
            if file_sha(base) != report["checkpoint_hashes"][category]:
                raise ValueError("Frozen category DQU checkpoint changed")
        while generator.poll() is None:
            guard()
            time.sleep(10)
        guard()
        if not all(file_sha(Path(name)) == sha for name, sha in source_hashes.items()):
            raise ValueError("Original FashionIQ metadata changed")
        save(status="complete", elapsed_seconds=time.monotonic() - started,
             interpretation="Final single-seed per-category results. No dataset expansion, no significance claim.")
    except BaseException as error:
        for child in children:
            stop_child(child)
        save(status="failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
