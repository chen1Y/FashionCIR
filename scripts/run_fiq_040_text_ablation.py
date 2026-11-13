"""Queue the 040 Flash text ablation behind the full FashionIQ 040H run."""

import argparse
import csv
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from dqu_flash_text_040 import file_sha256, load_queries, read_json
from run_fiq_040h_full_pair import NATIVE, choose_gpu
from run_fiq_context_generation_041 import write


RUN_NAME = "040_text_ablation"
WEIGHT = "open_clip_pytorch_model.bin"
CHECKPOINT_STAT = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
WAIT_FOR_EXIT = """
import os
from pathlib import Path
import select
import sys

pid, started = map(int, sys.argv[1:])
try:
    fd = os.pidfd_open(pid)
except ProcessLookupError:
    sys.exit(0)
try:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        sys.exit(0)
    if int(stat[19]) == started:
        poll = select.poll()
        poll.register(fd, select.POLLIN)
        poll.poll()
finally:
    os.close(fd)
"""


def file_identity(path):
    stat = path.stat()
    return {key: getattr(stat, key) for key in CHECKPOINT_STAT}


def prepare(root, predecessor, generation, data):
    control = read_json(predecessor / "preflight.json")
    texts = generation / "texts"
    inventory = read_json(texts / "inventory.json")
    source_state = read_json(generation / "state.json")
    generation_plan = read_json(generation / "plan.json")
    if (
        generation_plan["model"] != "qwen3.8-flash"
        or not generation_plan["settings"]["enable_thinking"]
        or source_state["status"] not in ("complete", "complete_with_unknown")
        or inventory["rows"] != 24016
        or inventory["totals"] != source_state["text_counts"]
        or file_sha256(texts / "inventory.json") != control["text_inventory_sha256"]
    ):
        raise ValueError("Flash source differs from the full 040H export")

    for name, digest in control["native_code"].items():
        if file_sha256(predecessor / "dqu_code" / name) != digest:
            raise ValueError(f"040H control source changed: {name}")
        if name != "dquTrain.py" and file_sha256(root / "dqu_code" / name) != digest:
            raise ValueError(f"Shared native DQU module changed: {name}")

    coverage = {}
    for category, (train, val, gallery) in NATIVE.items():
        for filename, digest in control["native_sources"][category].items():
            if file_sha256(Path(filename)) != digest:
                raise ValueError(f"Native DQU data changed: {filename}")
        coverage[category] = {}
        for split, count in (("train", train), ("val", val)):
            queries, statuses = load_queries(data, texts, category, split)
            if len(queries) != count:
                raise ValueError(f"Native DQU denominator changed: {category}/{split}")
            accepted = statuses.get("qwen_used", 0)
            coverage[category][split] = dict(total=count, flash=accepted, native_fallback=count - accepted,
                                              generation_statuses=statuses)
        coverage[category]["gallery"] = gallery

    checkpoint = (predecessor / "weights" / WEIGHT).resolve(strict=True)
    write(root / "plan.json", dict(
        experiment="040文本消融", native_control=str(predecessor), generation=str(generation),
        data=str(data), text_inventory_sha256=control["text_inventory_sha256"],
        generation_plan_sha256=file_sha256(generation / "plan.json"),
        input_coverage=coverage, api_calls=0,
        clip_checkpoint=str(checkpoint), clip_checkpoint_identity=file_identity(checkpoint),
        native_sources=control["native_sources"], native_control_train_sha256=control["native_code"]["dquTrain.py"],
        sealed_code={str(p.relative_to(root)): file_sha256(p)
                     for p in (Path(__file__), *(root / "dqu_code").glob("*.py"))},
        fallback="native BLIP-2 caption + corrected modifications",
        train_settings=dict(seed=42, batch_size=16, clip_lr=1e-6, fusion_lr=1e-4,
                            eval_every=1, patience=3, max_epochs=100),
    ))
    print(read_json(root / "plan.json")["input_coverage"])


def process_start(pid):
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


def wait_for_process(pid, started, executable, required_argument):
    if not pid or process_start(pid) != started:
        return
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
    except FileNotFoundError:
        return
    if process_start(pid) != started:
        return
    if cmdline != [""] and not {str(executable), required_argument} <= set(cmdline):
        raise ValueError(f"Unexpected 040H process command for PID {pid}")
    subprocess.run(["/usr/bin/python3", "-c", WAIT_FOR_EXIT, str(pid), str(started)], check=True)


def run(root, predecessor_pid, predecessor_start, child_pid, child_start):
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = root / "queue_state.json"
        if state.exists():
            raise FileExistsError("The text ablation queue has already started")
        plan = read_json(root / "plan.json")
        predecessor = Path(plan["native_control"])
        code = root / "dqu_code"
        generation = Path(plan["generation"])
        texts = generation / "texts"
        checkpoint = Path(plan["clip_checkpoint"])
        completed = {}
        identity = dict(queue_pid=os.getpid(), start_ticks=process_start(os.getpid()))
        write(state, dict(status="waiting_040h", **identity, predecessor_pid=predecessor_pid,
                          predecessor_start_ticks=predecessor_start))
        try:
            wait_for_process(predecessor_pid, predecessor_start,
                             predecessor / "code" / "run_fiq_040h_full_pair.py", str(predecessor))
            wait_for_process(child_pid, child_start, predecessor / "dqu_code" / "dquTrain.py", "040h_native")
            after = read_json(predecessor / "queue_state.json")
            identity["predecessor_status"] = after["status"]
            for filename, digest in plan["sealed_code"].items():
                if file_sha256(root / filename) != digest:
                    raise ValueError(f"Ablation code changed: {filename}")
            if (file_sha256(texts / "inventory.json") != plan["text_inventory_sha256"]
                    or file_sha256(generation / "plan.json") != plan["generation_plan_sha256"]
                    or file_identity(checkpoint) != plan["clip_checkpoint_identity"]):
                raise ValueError("Ablation input manifest or public OpenCLIP checkpoint changed")
            for category, (_, queries, gallery) in NATIVE.items():
                for filename, digest in plan["native_sources"][category].items():
                    if file_sha256(Path(filename)) != digest:
                        raise ValueError(f"Native DQU data changed: {filename}")
                card = choose_gpu(f"dqu_{category}", root, state, completed)
                if card is None:
                    write(state, dict(status="cancelled", **identity, completed=completed))
                    return
                gpu_index, gpu = card
                directory = root / category
                directory.mkdir()
                metrics = directory / "validation_metrics.csv"
                args = [sys.executable, "-u", str(code / "dquTrain.py"), "--dataset", category,
                        "--fashioniq-split", "original-split", "--fashioniq-path", plan["data"],
                        "--flash-text-dir", str(texts), "--clip-checkpoint", str(checkpoint),
                        "--seed", "42", "--batch-size", "16", "--num-workers", "4",
                        "--eval-every", "1", "--patience", "3", "--run-name", RUN_NAME,
                        "--model-dir", str(directory / "checkpoints"),
                        "--validation-metrics-csv", str(metrics),
                        "--gradient-checkpointing", "--adamw-no-foreach", "--memory-fraction", ".74"]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=f"{code}:/home/lcy/fashionCIR/src")
                with (directory / "train.log").open("wb") as log:
                    with subprocess.Popen(["nice", "-n", "8", *args], cwd="/home/lcy/fashionCIR",
                                          env=env, stdout=log, stderr=log) as child:
                        write(state, dict(status="running", **identity, arm=category, gpu=gpu_index,
                                          child_pid=child.pid, completed=completed, command=args))
                        completed[category] = child.wait()
                if completed[category] != 0:
                    raise RuntimeError(f"DQU training failed for {category}: exit {completed[category]}")
                with metrics.open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                if not rows or any(
                    (row["category"], int(row["queries"]), int(row["gallery"])) != (category, queries, gallery)
                    for row in rows
                ):
                    raise ValueError(f"Invalid validation coverage for {category}")
            write(state, dict(status="complete", **identity, completed=completed))
        except Exception as exc:
            write(state, dict(status="failed", **identity, completed=completed,
                              error=f"{type(exc).__name__}: {exc}"))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    action = parser.add_subparsers(dest="action", required=True)
    preflight = action.add_parser("prepare")
    preflight.add_argument("--predecessor", type=Path, required=True)
    preflight.add_argument("--generation", type=Path, required=True)
    preflight.add_argument("--data", type=Path, required=True)
    worker = action.add_parser("run")
    worker.add_argument("--predecessor-pid", type=int, default=0)
    worker.add_argument("--predecessor-start", type=int)
    worker.add_argument("--child-pid", type=int, default=0)
    worker.add_argument("--child-start", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.predecessor, args.generation, args.data)
    else:
        run(args.root, args.predecessor_pid, args.predecessor_start, args.child_pid, args.child_start)
