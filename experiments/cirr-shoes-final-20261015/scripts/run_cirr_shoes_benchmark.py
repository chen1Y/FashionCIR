"""Sequential original-DQU / structured / multi-positive experiments on one GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from benchmark_data import BenchmarkData, read_json, selection_score
from experiment_storage import require_space


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("cirr", "shoes"), required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--clip-checkpoint", required=True)
    p.add_argument("--qwen-model", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--baseline-epochs", type=int, default=100)
    p.add_argument("--adapter-epochs", type=int, default=6)
    p.add_argument("--generation-limit", type=int, default=0, help="0=all; nonzero is explicitly a limited-coverage pilot")
    p.add_argument("--min-free-gib", type=float, default=8)
    p.add_argument("--skip-baseline", action="store_true", help="Use a completed original-DQU checkpoint in this output directory")
    p.add_argument("--wait-for-baseline-pid", type=int, help="Linux: adopt an already running baseline before starting other GPU work")
    p.add_argument("--baseline-log", type=Path, help="Required with --wait-for-baseline-pid to verify successful completion")
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.wait_for_baseline_pid and (not args.skip_baseline or not args.baseline_log):
        p.error("--wait-for-baseline-pid requires --skip-baseline and --baseline-log")
    args.output = args.output.resolve()
    data = BenchmarkData(args.data_root, args.dataset)
    checkpoint_dir = args.output / "checkpoints"
    base = checkpoint_dir / f"{args.dataset}_dqu_original_seed{args.seed}_best.pt"
    shared = ["--dataset", args.dataset, f"--{args.dataset}-path", str(Path(args.data_root).resolve()),
              "--clip-checkpoint", args.clip_checkpoint, "--model-dir", str(checkpoint_dir),
              "--seed", str(args.seed), "--min-free-gib", str(args.min_free_gib)]
    commands = []
    if not args.skip_baseline:
        commands.append(("dqu_original", [sys.executable, "src/dquTrain.py", *shared,
            "--lr", "5e-5" if args.dataset == "shoes" else "1e-4", "--clip-lr",
            "5e-6" if args.dataset == "shoes" else "1e-6", "--run-name", "dqu_original",
            "--num-epochs", str(args.baseline_epochs), "--patience", "5"]))
    structured = {split: args.output / f"structured_{args.dataset}_{split}.json" for split in ("train", "val")}
    if not args.skip_generation:
        for split in structured:
            commands.append((f"qwen_{split}", [sys.executable, "src/generate_benchmark_edits.py",
                "--dataset", args.dataset, "--root", str(Path(args.data_root).resolve()),
                "--split", split, "--model", args.qwen_model, "--offline", "--limit",
                str(args.generation_limit), "--output", str(structured[split])]))
    for mode in ("off", "identity", "annotated"):
        commands.append((f"structured_{mode}", [sys.executable, "src/newTrain.py", *shared,
            "--structured-train-path", str(structured["train"]), "--structured-val-path", str(structured["val"]),
            "--dqu-checkpoint", str(base), "--freeze-dqu-backbone", "--multi-positive", mode,
            "--hard-negative-weight", "0", "--num-epochs", str(args.adapter_epochs),
            "--patience", str(args.adapter_epochs+1), "--run-name", f"structured_{mode}"]))
    report = {"dataset_audit": data.audit(), "config": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}, "commands": commands,
              "code_sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in sorted((ROOT/"src").glob("*.py"))
                  if f.name not in ("generateModify.py", "generatetext.py", "generatetextbyimg.py")},
              "stages": [], "status": "planned", "generation_scope": "full" if args.generation_limit == 0 else "limited_coverage_pilot"}
    report["config"]["output"] = str(args.output)
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return
    if not data.audit()["ready"]:
        raise FileNotFoundError("Dataset images missing; no GPU stages started")
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "experiment_report.json"
    def save():
        report_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    if args.wait_for_baseline_pid:
        report["status"] = "waiting_for_original_dqu"
        save()
        process = Path(f"/proc/{args.wait_for_baseline_pid}")
        while process.exists():
            try:
                if process.joinpath("stat").read_text().split(") ", 1)[1].split()[0] == "Z":
                    break
                command = process.joinpath("cmdline").read_bytes().replace(b"\0", b" ")
                if b"dquTrain.py" not in command or (b"--dataset " + args.dataset.encode()) not in command:
                    raise RuntimeError("Baseline PID identity changed; refusing to start queued GPU jobs")
            except FileNotFoundError:
                break
            require_space(args.output, args.min_free_gib)
            time.sleep(30)
        text = args.baseline_log.read_text()
        if "early stopping after" not in text and f"epoch={args.baseline_epochs} metrics=" not in text:
            report["status"] = "stopped_baseline_completion_unverified"
            save()
            raise RuntimeError("Original DQU did not finish successfully; queued stages stopped")
    if args.skip_baseline and (not base.is_file() or not base.with_suffix(".json").is_file()):
        raise FileNotFoundError("Completed baseline checkpoint and metrics are required")
    for name, command in commands:
        require_space(args.output, args.min_free_gib)
        report["status"] = name
        save()
        with (args.output/f"{name}.log").open("w", encoding="utf-8") as log:
            code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        report["stages"].append({"stage": name, "returncode": code})
        save()
        if code:
            report["status"] = "stopped_on_error"
            save()
            raise RuntimeError(f"{name} failed; see {args.output / (name+'.log')}")
    metrics = {"dqu_original": read_json(base.with_suffix(".json"))}
    for mode in ("off", "identity", "annotated"):
        metrics[f"structured_{mode}"] = read_json(checkpoint_dir /
            f"{args.dataset}_structured_{mode}_seed{args.seed}_best.json")
    original_score = selection_score(metrics["dqu_original"], args.dataset)
    report.update(status="complete", metrics=metrics,
        selection_deltas={name: selection_score(values, args.dataset)-original_score
                          for name, values in metrics.items()})
    report["structured_coverage"] = {s: sum(x["status"] == "ok" for x in read_json(f)["samples"])/len(data.queries[s])
                                     for s, f in structured.items()}
    save()


if __name__ == "__main__":
    main()
