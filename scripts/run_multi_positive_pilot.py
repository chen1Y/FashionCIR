"""Run sequential, budget-matched FashionIQ ablations; never change evaluation labels."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fashioniq-path", required=True)
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--dqu-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="dress", choices=("dress", "shirt", "toptee"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--modes", nargs="+", choices=("off", "identity", "annotated"),
                        default=["off", "annotated"])
    parser.add_argument("--num-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite a previous pilot or accidentally mix its results.
    status_path = output / "pilot_status.json"
    if status_path.exists():
        raise FileExistsError(status_path)
    status = {"started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "config": vars(args), "code_sha256": {
                  name: hashlib.sha256((root / "src" / name).read_bytes()).hexdigest()
                  for name in ("newTrain.py", "newModel.py", "newDataset.py", "multi_positive.py", "multi_positive_loss.py")},
              "runs": []}
    def save():
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temporary.replace(status_path)
    save()
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for seed in args.seeds:
        for mode in args.modes:
            name = f"mp_pilot_{mode}"
            command = [sys.executable, "newTrain.py", "--dataset", args.dataset,
                "--fashioniq-split", "original-split", "--fashioniq-path", str(Path(args.fashioniq_path).resolve()),
                "--clip-checkpoint", str(Path(args.clip_checkpoint).resolve()),
                "--dqu-checkpoint", str(Path(args.dqu_checkpoint).resolve()),
                "--model-dir", str(output / "checkpoints"), "--run-name", name,
                "--multi-positive", mode, "--seed", str(seed), "--num-epochs", str(args.num_epochs),
                "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers),
                "--patience", "20", "--hard-negative-weight", "0"]
            # All modes share v7's two-epoch gate warmup, optimizer, structured
            # JSON, DQU initializer, seed and full-split evaluation. Hard-margin
            # weight 0 matches v7 and isolates the multi-positive intervention.
            log = output / f"{mode}_seed{seed}.log"
            record = {"mode": mode, "seed": seed, "status": "running", "command": command,
                      "log": str(log), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
            status["runs"].append(record)
            save()
            print(json.dumps({"event": "start", "mode": mode, "seed": seed}), flush=True)
            start = time.monotonic()
            with log.open("x", encoding="utf-8") as handle:
                result = subprocess.run(command, cwd=root / "src", env=env, stdout=handle, stderr=subprocess.STDOUT)
            record.update(returncode=result.returncode, seconds=time.monotonic() - start,
                          status="complete" if result.returncode == 0 else "failed")
            metrics_path = output / "checkpoints" / f"{args.dataset}_{name}_seed{seed}_best.json"
            if result.returncode == 0:
                record["best_metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
            save()
            print(json.dumps(record), flush=True)
            if result.returncode:
                raise SystemExit(result.returncode)
    status["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    save()


if __name__ == "__main__":
    main()
