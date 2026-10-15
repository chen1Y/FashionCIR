"""Wait for all Shoes stages, then label a bounded FashionIQ batch on the idle GPU."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experiment_storage import require_space
from fashioniq_augmentation import build_overlay, propose, validate, write_json
from fashioniq_noise import read_json

STAGES = {"qwen_train", "qwen_val", "structured_off", "structured_identity", "structured_annotated"}
MODES = {"dqu_original", "structured_off", "structured_identity", "structured_annotated"}


def shoes_state(report):
    """A stage name, dead PID or baseline completion alone is never sufficient."""
    status = report.get("status", "unknown")
    if status.startswith("stopped") or any(s.get("returncode") != 0 for s in report.get("stages", [])):
        return "failed"
    if status != "complete":
        return "waiting"
    completed = {s["stage"] for s in report.get("stages", []) if s.get("returncode") == 0}
    metrics = report.get("metrics", {})
    if not STAGES <= completed or not MODES <= metrics.keys():
        raise ValueError("Incomplete Shoes success evidence")
    for mode in MODES:
        for key in ("shoes_r1", "shoes_r10", "shoes_r50"):
            value = metrics[mode].get(key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError("Invalid Shoes metrics")
    return "ready"


def merge_proposals(pilot, proposed, attempts):
    """Keep the reviewed pilot; spend no labels on its pairs or degree-cap conflicts."""
    if attempts < 1 or pilot["source_signature"] != proposed["source_signature"] or pilot["category"] != proposed["category"]:
        raise ValueError("Invalid proposal budget or incompatible pilot")
    result = copy.deepcopy(proposed)
    result["records"] = copy.deepcopy(pilot["records"])
    seen = {(r["candidate"], r["target"]) for r in result["records"]}
    accepted = [r for r in result["records"] if r.get("decision") == "accept"]
    refs = Counter(r["candidate"] for r in accepted)
    targets = Counter(r["target"] for r in accepted)
    count = 0
    for row in proposed["records"]:
        pair = row["candidate"], row["target"]
        if pair in seen or refs[pair[0]] >= 2 or targets[pair[1]] >= 2:
            continue
        result["records"].append(copy.deepcopy(row))
        seen.add(pair)
        refs[pair[0]] += 1
        targets[pair[1]] += 1
        count += 1
        if count >= attempts:
            break
    if count != attempts:
        raise ValueError(f"Only {count} eligible new pairs; requested {attempts}")
    result.update(review_method="Prior Codex visual-reviewed pilot plus new two-prompt Qwen checks using the same model; synthetic labels, not human GT",
                  prior_accepted=len(accepted), new_attempt_budget=count)
    return result


def active_benchmark(results):
    for name in ("baseline.pid", "queue.pid"):
        path = Path(results) / name
        if not path.exists():
            continue
        pid = int(path.read_text().strip())
        process = Path(f"/proc/{pid}")
        try:
            if process.joinpath("stat").read_text().split(") ", 1)[1].split()[0] == "Z":
                continue
            cmd = process.joinpath("cmdline").read_bytes().replace(b"\0", b" ")
        except FileNotFoundError:
            continue
        expected = b"dquTrain.py" if name == "baseline.pid" else b"run_cirr_shoes_benchmark.py"
        if expected not in cmd or b"--dataset shoes" not in cmd:
            raise RuntimeError("Benchmark PID identity changed; refusing automatic follow-up")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shoes-results", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--noise-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--new-attempts", type=int, default=120)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--min-free-gib", type=float, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.new_attempts <= 240 or args.min_free_gib < 8:
        parser.error("new-attempts must be 1..240; disk reserve >=8 GiB")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    # Linux lock held across waiting, generation and export. No duplicate GPU launch.
    import fcntl
    with (args.output / "followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = args.output / "followup_state.json"
        prepared = args.output / "prepared_proposals.json"
        reviewed = args.output / "qwen_reviewed.json"
        if state_path.exists() and read_json(state_path).get("status") not in {"prepared", "waiting_for_shoes", "waiting_for_gpu"}:
            raise RuntimeError("Prior run needs review; outputs will not be overwritten or retried automatically")
        state = {"schema_version": 1, "new_attempt_budget": args.new_attempts, "seed": args.seed,
                 "model": args.model, "shoes_results": str(args.shoes_results),
                 "code_root": str(ROOT), "new_images": 0}

        def save(status, **extra):
            state.update(status=status, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
            write_json(state_path, state)
            print(json.dumps(state), flush=True)

        try:
            require_space(args.output, args.min_free_gib)
            pilot = read_json(args.pilot)
            validate(args.root, pilot)
            proposed = propose(args.root, pilot["category"], limit=args.new_attempts * 2,
                               seed=args.seed, noise_audit=args.noise_audit)
            manifest = merge_proposals(pilot, proposed, args.new_attempts)
            if prepared.exists() and read_json(prepared) != manifest:
                raise RuntimeError("Prepared proposals/config changed; use a new output directory")
            write_json(prepared, manifest)
            save("prepared", prior_accepted=manifest["prior_accepted"], source_signature=manifest["source_signature"])
            if args.prepare_only:
                return
            previous = None
            while True:
                require_space(args.output, args.min_free_gib)
                try:
                    report = read_json(args.shoes_results / "experiment_report.json")
                except json.JSONDecodeError:
                    time.sleep(30)  # Existing queue writes this file non-atomically.
                    continue
                ready = shoes_state(report)
                if ready == "failed":
                    raise RuntimeError("Shoes stopped/failed; follow-up held without starting generation")
                if ready != "ready" or active_benchmark(args.shoes_results):
                    if previous != "waiting_for_shoes":
                        save("waiting_for_shoes")
                        previous = "waiting_for_shoes"
                    time.sleep(30)
                    continue
                gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
                if gpu:
                    if previous != "waiting_for_gpu":
                        save("waiting_for_gpu")
                        previous = "waiting_for_gpu"
                    time.sleep(30)
                    continue
                break
            save("labeling")
            command = [sys.executable, "src/generate_fashioniq_pair_captions.py", "--root", str(args.root),
                       "--manifest", str(prepared), "--output", str(reviewed), "--model", args.model,
                       "--guard-results", str(args.shoes_results), "--min-free-gib", str(args.min_free_gib),
                       "--limit", str(args.new_attempts)]
            with (args.output / "labeling.log").open("x") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            require_space(args.output, args.min_free_gib)
            labeled = read_json(reviewed)
            counts = Counter(r.get("decision") for r in labeled["records"])
            new_accepted = counts["accept"] - manifest["prior_accepted"]
            if new_accepted < 1:
                raise RuntimeError("No new pairs passed review; no replacement overlay exported")
            summary = build_overlay(args.root, labeled, args.output / "overlay")
            save("complete", new_accepted=new_accepted, decisions=dict(counts), summary=summary,
                 note="Synthetic labels; retrieval improvement untested. No retrieval training started.")
        except Exception as exc:
            save("stopped", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
