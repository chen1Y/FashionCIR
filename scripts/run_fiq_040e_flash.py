"""Generate the matched 040C rows with Flash, then run the sealed 040E training."""

import argparse
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from zipfile import ZipFile

from run_fiq_context_generation_040c import TOTAL, main as generate
from run_fiq_context_generation_041 import SETTINGS, write


MODEL = "qwen3.8-flash"
LABEL = "040E（Flash）"
PROMPT = "fiq_context_generation_041_prompt.txt"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def options(root, dry_run=False, resume=False):
    return SimpleNamespace(bundle=root / "reference_inputs.zip", prompt=root / "code" / PROMPT,
                           output=root / "generation.json", dry_run=dry_run, resume=resume)


def preflight(root, prior):
    plan = read(prior / "plan.json")
    pilot = plan["pilot"]
    cases = root / "cases.json"
    if (plan["arms"] != ["040E"] or plan["experiment"] != "040E" or pilot["model"] != "qwen3.7-plus"
            or pilot["generation_settings"] != SETTINGS or pilot["max_requests"] != TOTAL):
        raise ValueError("Prior 040E plan differs from the expected control")
    if sha(cases) != pilot["cases_sha256"] or sha(root / "code" / PROMPT) != pilot["prompt_sha256"]:
        raise ValueError("040C sample manifest or 041 prompt changed")
    with ZipFile(root / "reference_inputs.zip") as bundle:
        if bundle.read("cases.json") != cases.read_bytes():
            raise ValueError("Reference archive has a different case manifest")
    if sha(prior / "generation.json") != pilot["generation_sha256"]:
        raise ValueError("Prior Plus responses changed")
    for name, digest in plan["source_sha256"].items():
        if name == "run_fiq_context_generation_040c.py":
            continue  # The generation runner now accepts the model explicitly.
        if sha(root / "code" / name) != digest:
            raise ValueError(f"040E training source differs: {name}")
    for category, item in plan["inputs"].items():
        if sha(root / "metadata" / f"{category}.json") != item["metadata_sha256"]:
            raise ValueError(f"040E metadata differs: {category}")
    if sha(root / "texts" / "inventory.json") != plan["text_inventory_sha256"]:
        raise ValueError("040E fallback texts changed")
    if (root / "weights" / "open_clip_pytorch_model.bin").resolve() != (
            prior / "weights" / "open_clip_pytorch_model.bin").resolve():
        raise ValueError("040E initialization checkpoint changed")
    generate(options(root, dry_run=True), model=MODEL)
    sources = {p.name: sha(p) for p in (root / "code").iterdir() if p.is_file()}
    prepared = dict(label=LABEL, model=MODEL, settings=SETTINGS, requests=TOTAL, workers=24,
                    control=str(prior), control_plan_sha256=sha(prior / "plan.json"),
                    bundle_sha256=sha(root / "reference_inputs.zip"), source_sha256=sources)
    write(root / "preflight.json", prepared)


def gpu_snapshot():
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid",
                                         "--format=csv,noheader"], text=True)
    busy = {line.strip() for line in processes.splitlines() if line.strip()}
    cards = {}
    for line in rows.splitlines():
        index, gpu, free, usage = (field.strip() for field in line.split(","))
        if int(free) >= 22000 and int(usage) <= 10 and gpu not in busy:
            cards[int(index)] = gpu
    return cards


def train(root, prepared, generation):
    prior = Path(prepared["control"])
    plan_path = prior / "plan.json"
    if sha(plan_path) != prepared["control_plan_sha256"]:
        raise ValueError("Original 040E plan changed")
    plan = read(plan_path)
    first = gpu_snapshot()
    time.sleep(5)
    second = gpu_snapshot()
    gpu_index = next((i for i in (plan["gpu_index"], 3, 1, 0, 2)
                      if i in first and first[i] == second.get(i)), None)
    if gpu_index is None:
        write(root / "experiment_state.json", dict(status="generation_complete_waiting_gpu",
                                                   generation=generation, first=first, second=second))
        return
    gpu = second[gpu_index]
    plan["pilot"].update(model=MODEL, generation_sha256=sha(root / "generation.json"))
    plan.update(experiment=LABEL, control="040E", gpu=gpu, gpu_index=gpu_index,
                caption_source=dict(model=MODEL, control_plan_sha256=prepared["control_plan_sha256"]),
                source_sha256=prepared["source_sha256"])
    write(root / "plan.json", plan)
    write(root / "experiment_state.json", dict(status="training", gpu_index=gpu_index, generation=generation))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
               PYTHONPATH=f"{root / 'code'}:/home/lcy/fashionCIR/src")
    with (root / "train.log").open("wb") as log:
        command = ["nice", "-n", "8", sys.executable, "-u",
                   str(root / "code" / "run_fiq_vit_lora_039a_earlystop.py"), "--root", str(root), "--gpu", gpu]
        subprocess.run(command, cwd="/home/lcy/fashionCIR", env=env, check=True, stdout=log, stderr=log)
    summary = read(root / "summary.json")
    if read(root / "state.json")["status"] != "complete" or summary["last"]["stale"] != 3:
        raise ValueError("040E training did not complete normal early stopping")
    write(root / "experiment_state.json", dict(status="complete", gpu_index=gpu_index,
                                               generation=generation, training=summary))


def work(root, prepared, key, resume, train_only):
    if not train_only:
        write(root / "experiment_state.json", dict(status="generating", pid=os.getpid(), model=MODEL))
        if generate(options(root, resume=resume), model=MODEL, key=key,
                    workers=prepared["workers"], confirm_access=True):
            state = read(root / "generation.json")
            errors = [a for a in state["answers"] if a.get("error")]
            write(root / "experiment_state.json", dict(
                status="generation_failed", submitted=state["submitted"], known_responses=len(state["answers"])-len(errors),
                errors=[dict(error=a["error"], provider_code=a.get("provider_code")) for a in errors]))
            return
    del key
    state = read(root / "generation.json")
    answers = state["answers"]
    if state["status"] != "complete" or len(answers) != TOTAL or any("error" in a for a in answers):
        raise ValueError("Incomplete Flash generation; training cannot start")
    usage = [a["usage"] for a in answers if a.get("usage")]
    input_tokens = sum(x["prompt_tokens"] for x in usage)
    output_tokens = sum(x["completion_tokens"] for x in usage)
    generation = dict(model=MODEL, rows=len(answers), thinking=sum(a["thinking_present"] for a in answers),
                      program_pass=sum(a["thinking_present"] and a["finish"] == "stop" and not a["flags"] for a in answers),
                      input_tokens=input_tokens, output_tokens=output_tokens, usage_missing=TOTAL-len(usage),
                      estimated_cny=(input_tokens * .8 + output_tokens * 2.7) / 1e6)
    train(root, prepared, generation)


def launch(root, resume, train_only):
    with (root / "experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepared = read(root / "preflight.json")
        for name, digest in prepared["source_sha256"].items():
            if sha(root / "code" / name) != digest:
                raise ValueError(f"Sealed source changed: {name}")
        if sha(root / "reference_inputs.zip") != prepared["bundle_sha256"]:
            raise ValueError("Sealed reference archive changed")
        if (root / "state.json").exists() or (not resume and not train_only and (root / "generation.json").exists()):
            raise FileExistsError("040E Flash already has an active or existing stage")
        if not train_only and not sys.stdin.isatty():
            raise ValueError("An interactive terminal is required for hidden input")
        key = None if train_only else getpass.getpass("DashScope API key (hidden input): ").strip()
        if not train_only and not key:
            raise ValueError("API key is empty")
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        pid = os.fork()
        if pid:
            print(f"040E Flash worker started, PID {pid}.", flush=True)
            return
        os.setsid()
        with open(os.devnull, "rb") as stdin, (root / "worker.log").open("ab", buffering=0) as log:
            os.dup2(stdin.fileno(), 0)
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            try:
                work(root, prepared, key, resume, train_only)
            except Exception as error:
                write(root / "experiment_state.json", dict(status="failed", error=type(error).__name__))
                raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-from", type=Path)
    action.add_argument("--launch", action="store_true")
    action.add_argument("--resume", action="store_true")
    action.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_from:
        preflight(args.root, args.prepare_from)
    else:
        launch(args.root, args.resume, args.train_only)
