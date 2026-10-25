"""Adopt a durable user pause on the new server; never retrain completed stages."""
from __future__ import annotations

import argparse
import copy
import datetime
import getpass
import json
import math
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
from generate_fashioniq_online import EXPECTED, check_complete, fingerprint
from run_fiq_shirt_toptee import publish_checkpoint
from run_cirr_shoes_benchmark import detach_queue


def remaining_commands(protocol, report, resume_plan):
    completed = {(r["category"], r["stage"]) for r in report["stages"]
                 if r["status"] == "complete" and r.get("returncode") == 0}
    if completed != {tuple(r) for r in resume_plan["completed_stages"]}:
        raise ValueError("Saved completed-stage records disagree")
    pending = [dict(category=category, stage=name, command=command)
               for category, stages in protocol["commands"].items() for name, command in stages
               if (category, name) not in completed]
    if pending != resume_plan["remaining_gpu_commands"]:
        raise ValueError("Saved continuation commands change the original protocol")
    return pending


def remaining_budget(audit):
    remaining = float(audit["remaining_original_wall_budget_seconds"])
    spent = float(audit["spent_wall_seconds_before_pause"])
    if not all(math.isfinite(x) for x in (remaining, spent)) or spent < 0 or remaining < 0:
        raise ValueError("Invalid saved runtime budget")
    if abs(spent + remaining - 43200) > .01:
        raise ValueError("Continuation would reset or increase original runtime budget")
    return remaining


def check_ledger(path, category, rows_by_split):
    import sqlite3
    db = sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True)
    try:
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("API ledger integrity failed")
        config = json.loads(db.execute("SELECT value FROM config WHERE id=1").fetchone()[0])
        if (config["category"] != category or config["model"] != "qwen3.7-flash"
                or config["max_cny"] != 7.5 or config["max_calls"] != 2 * sum(EXPECTED[category].values())
                or config["fingerprints"] != {s:fingerprint(rs) for s,rs in rows_by_split.items()}):
            raise ValueError("API ledger model, budget or input identity changed")
        if db.execute("SELECT COUNT(*) FROM requests WHERE state IN ('reserved','metered')").fetchone()[0]:
            raise ValueError("Pause has unresolved in-flight requests; inspect before resume")
        for split, idx, encoded in db.execute("SELECT split,idx,value FROM samples"):
            sample = json.loads(encoded)
            row = rows_by_split[split][idx]
            if sample["source_index"] != idx or any(sample.get(k) != row[k] for k in (
                    "candidate", "target", "modification_text", "category", "split")):
                raise ValueError("Saved API sample binding changed")
        count, charge = db.execute("SELECT COUNT(*),COALESCE(SUM(charge),0) FROM requests").fetchone()
        if count >= config["max_calls"] or charge >= config["max_cny"]:
            raise ValueError("Saved API category budget is exhausted")
        return dict(samples=db.execute("SELECT COUNT(*) FROM samples").fetchone()[0],
                    calls=count, accounted_upper_cny=charge)
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    root = Path("/root/autodl-tmp/data/FashionIQ")
    protocol = json.loads((output / "protocol.json").read_text())
    report = json.loads((output / "experiment_report.json").read_text())
    audit = json.loads((output / "pause_audit.json").read_text())
    resume_plan = json.loads((output / "pause_resume_plan.json").read_text())
    if report["status"] != "paused_by_user" or audit["status"] != "paused_by_user":
        raise ValueError("This entry point only adopts the inspected durable user pause")
    commands = remaining_commands(protocol, report, resume_plan)
    available = remaining_budget(audit)
    if len(commands) != 7 or commands[0]["stage"] != "structured_off":
        raise ValueError("Unexpected remaining-stage scope; inspect explicitly")
    for name, expected in protocol["source_hashes"].items():
        if file_sha(Path(name)) != expected:
            raise ValueError("Original dataset metadata changed: " + name)
    for name, expected in protocol["code_hashes"].items():
        if file_sha(ROOT / name) != expected:
            raise ValueError("Original experiment source changed: " + name)
    checkpoint = Path(audit["checkpoint_path"])
    if file_sha(checkpoint) != audit["checkpoint_sha256"] or report["checkpoint_hashes"]["shirt"] != audit["checkpoint_sha256"]:
        raise ValueError("Completed Shirt checkpoint identity changed")
    basemetric = json.loads(checkpoint.with_suffix(".json").read_text())
    if basemetric != report["metrics"]["shirt"]["dqu_original"]:
        raise ValueError("Completed Shirt metrics changed")
    # Verify the pretrained asset referenced by the saved commands, without downloads.
    command0 = commands[0]["command"]
    clip_path = Path(command0[command0.index("--clip-checkpoint") + 1])
    if file_sha(clip_path) != protocol["clip_sha256"]:
        raise ValueError("Pretrained CLIP asset changed")
    ledgers = {}
    for category in ("shirt", "toptee"):
        rows = {s:load_fashioniq_queries(root, category, s) for s in ("train", "val")}
        if {s:len(rs) for s,rs in rows.items()} != EXPECTED[category]:
            raise ValueError("Original full split sizes changed")
        for split in ("train", "val"):
            image_ids = json.loads((root / "image_splits" / f"split.{category}.{split}.json").read_text())
            if not all((root / "resized_image" / f"{i}.jpg").is_file() for i in image_ids):
                raise FileNotFoundError("Incomplete cloned training/gallery images")
        ledger = output / category / "requests.sqlite3"
        if ledger.exists():
            ledgers[category] = check_ledger(ledger, category, rows)
    if ledgers.get("shirt", {}).get("samples") != audit["api_final_samples"]:
        raise ValueError("Pause generation count changed")
    if (output / "continuation_budget_20261025.json").exists():
        raise FileExistsError("Continuation already exists; inspect it, never reset its clock")
    need_new_base = checkpoint.stat().st_size / 2**30
    # Includes the remaining durable Toptee baseline and a conservative JSON/log allowance.
    free = require_space(output, 8 + need_new_base + .5)
    require_space(Path("/dev/shm"), 16)
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("GPU occupied; do not overlap experiments")
    preflight = dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        completed_shirt_sha256=audit["checkpoint_sha256"], reused_api_records=audit["api_final_samples"],
        remaining_queries=16014-audit["api_final_samples"], remaining_gpu_stages=len(commands),
        remaining_runtime_seconds=available, free_gib=free, existing_ledgers=ledgers,
        log_offsets_before_resume={name:(output / name).stat().st_size for name in ("queue.log", "online_generation.log")},
        resume_script_sha256=file_sha(Path(__file__)), original_scientific_protocol_unchanged=True,
        runtime_policy="Explicit user continuation excludes the requested pause; consumed runtime and all API costs retained")
    if args.dry_run:
        print(json.dumps(preflight, indent=2))
        return
    import fcntl
    queue_lock = (output / "queue.lock").open("a")
    fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    key = getpass.getpass("DashScope API key (memory only): ")
    if not key:
        raise ValueError("Empty API key")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = output / ("continuation_" + stamp)
    backup.mkdir()
    for name in ("experiment_report.json", "cost_summary.json", "pause_audit.json", "pause_resume_plan.json", "queue.pid", "generation.pid"):
        source = output / name
        if source.exists():
            shutil.copyfile(source, backup / name)
    atomic_write_json(backup / "resume_preflight.json", preflight)
    if args.detach and not detach_queue(output):
        return
    started = time.monotonic()
    budget = dict(original_maximum_seconds=43200, previously_spent_seconds=audit["spent_wall_seconds_before_pause"],
                  maximum_continuation_seconds=available, resumed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  excluded_user_pause=True, original_started_utc=report["started_utc"], budget_reset=False)
    atomic_write_json(output / "continuation_budget_20261025.json", budget)
    atomic_write_json(output / "resume_preflight_20261025.json", preflight)
    (output / "queue.pid").write_text(str(os.getpid()) + "\n")
    report = copy.deepcopy(report)
    report.update(status="resuming_saved_pause", queue_pid=os.getpid(), active_dispatcher=Path(__file__).name,
                  continuation=budget, previous_pause="pause_audit.json", resumed_from=str(backup))
    report.pop("error", None)
    report.pop("error_type", None)
    children = []
    generator = None
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")

    def save(**values):
        report.update(values)
        report["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cost_path = output / "cost_summary.json"
        if cost_path.exists():
            report["costs"] = json.loads(cost_path.read_text())
        atomic_write_json(output / "experiment_report.json", report)

    def guard():
        require_space(output)
        if time.monotonic() - started > available:
            raise TimeoutError("Original remaining runtime budget exhausted; no automatic extension")
        if generator is not None and generator.poll() not in (None, 0):
            raise RuntimeError("Online generation failed; preserve responses and inspect original ledger")

    def stop(child):
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()

    try:
        read_fd, write_fd = os.pipe()
        with (output / "online_generation.log").open("a") as log:
            log.write("\n=== Explicit saved-pause continuation " + budget["resumed_utc"] + " ===\n")
            log.flush()
            try:
                generator = subprocess.Popen([sys.executable, "src/generate_fashioniq_online.py", "--root", str(root),
                    "--output", str(output), "--api-key-fd", str(read_fd)], cwd=ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, pass_fds=(read_fd,), start_new_session=True)
            finally:
                os.close(read_fd)
            children.append(generator)
            try:
                os.write(write_fd, key.encode())
            finally:
                os.close(write_fd)
            key = ""
        save(generation_pid=generator.pid, status="waiting_online_shirt")
        for step in commands:
            category, name, command = step["category"], step["stage"], step["command"]
            folder = output / category
            folder.mkdir(exist_ok=True)
            checkpoint_dir = folder / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            report["metrics"].setdefault(category, {})
            guard()
            if name.startswith("structured_"):
                save(status="waiting_online_" + category)
                while True:
                    ready = all((folder / f"structured_{category}_{split}.json").exists()
                        and json.loads((folder / f"structured_{category}_{split}.json").read_text()).get("state") == "complete"
                        for split in ("train", "val"))
                    if ready:
                        break
                    guard()
                    if generator.poll() == 0:
                        raise ValueError("Generator exited without full split records")
                    time.sleep(10)
                for split in ("train", "val"):
                    check_complete(json.loads((folder / f"structured_{category}_{split}.json").read_text()),
                                   load_fashioniq_queries(root, category, split), category, split)
            else:
                staging = Path(command[command.index("--model-dir") + 1])
                if staging.parent != Path("/dev/shm/fiq_shirt_toptee_20261022") or staging.exists():
                    raise ValueError("Inspect existing/mismatched baseline staging; no overwrite")
                staging.parent.mkdir(exist_ok=True)
                require_space(Path("/dev/shm"), 16)
            if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
                raise RuntimeError("GPU occupied before sequential stage")
            log_path = folder / (name + ".log")
            row = dict(category=category, stage=name, command=command, status="running",
                       started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            report["stages"].append(row)
            with log_path.open("x") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(child)
                row["pid"] = child.pid
                save(status=category + "_" + name)
                while child.poll() is None:
                    guard()
                    with log_path.open("rb") as stream:
                        stream.seek(max(0, log_path.stat().st_size - 10000))
                        tail = stream.read().decode(errors="replace")
                    if re.search(r"(?:loss|train_loss)=(?:nan|inf)\b|CUDA out of memory|OutOfMemoryError", tail, re.I):
                        raise RuntimeError("Non-finite loss/OOM; stop remaining stages")
                    time.sleep(10)
            row.update(returncode=child.returncode, status="complete" if child.returncode == 0 else "failed")
            save()
            if child.returncode:
                raise RuntimeError(f"{category}/{name} failed")
            if name == "dqu_original":
                source = staging / f"{category}_dqu_original_seed42_best.pt"
                destination = checkpoint_dir / source.name
                report["checkpoint_hashes"][category] = publish_checkpoint(source, destination)
                source.unlink()
                source.with_suffix(".json").unlink()
            metric_path = checkpoint_dir / f"{category}_{name}_seed42_best.json"
            metric = json.loads(metric_path.read_text())
            if not all(math.isfinite(metric[k]) and 0 <= metric[k] <= 100
                       for k in (category + "_r1", category + "_r10", category + "_r50")):
                raise ValueError("Final checkpoint metrics are invalid")
            report["metrics"][category][name] = metric
            save()
        while generator.poll() is None:
            guard()
            time.sleep(10)
        guard()
        for category, expected in report["checkpoint_hashes"].items():
            path = output / category / "checkpoints" / f"{category}_dqu_original_seed42_best.pt"
            if file_sha(path) != expected:
                raise ValueError("Frozen original DQU checkpoint changed")
        if not all(file_sha(Path(name)) == h for name,h in protocol["source_hashes"].items()):
            raise ValueError("Original metadata changed")
        save(status="complete", continuation_elapsed_seconds=time.monotonic() - started,
             total_consumed_runtime_seconds=audit["spent_wall_seconds_before_pause"] + time.monotonic() - started,
             interpretation="Final per-category single-seed DQU/off/identity/annotated. No data expansion or significance claim.")
    except BaseException as error:
        for child in children:
            stop(child)
        save(status="failed", error_type=type(error).__name__, error=str(error),
             continuation_elapsed_seconds=time.monotonic() - started)
        raise


if __name__ == "__main__":
    main()
