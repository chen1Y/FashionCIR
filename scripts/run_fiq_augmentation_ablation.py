"""Bounded v7 data ablation: original prefix vs all screened Dress triplets."""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fashioniq_augmentation import write_json, validate
from fashioniq_noise import file_sha
from experiment_storage import require_space


def valid(sample):
    return (sample.get("status") == "ok" and sample.get("validation", {}).get("valid") is True
            and sample.get("semantic_validation", {}).get("valid") is True)


def merge_structured(original, extra, expected_pairs):
    """Preserve every original record, with disjoint, query-only extra records."""
    if not isinstance(original, dict) or not isinstance(extra, dict):
        raise ValueError("Structured files must be metadata containers")
    old = original["samples"]
    new = extra["samples"]
    keys = [(s["candidate"], s["target"]) for s in new]
    if len(keys) != len(set(keys)) or set(keys) != set(expected_pairs):
        raise ValueError("New structures do not cover the locked 101 pairs exactly")
    if set(keys) & {(s["candidate"], s["target"]) for s in old}:
        raise ValueError("New structure collides with original training records")
    for row in new:
        if row.get("category") != "dress" or row.get("split") != "train":
            raise ValueError("Unexpected category/split in new structures")
        if row.get("generation_input", {}).get("target_was_provided_to_model") is not False:
            raise ValueError("Query-only generation proof is missing")
    fraction = sum(valid(s) for s in new) / len(new)
    if fraction < .9:
        raise ValueError(f"Only {fraction:.1%} validated new structures; requires 90%")
    result = copy.deepcopy(original)
    result["samples"] = copy.deepcopy(old) + copy.deepcopy(new)
    result["augmentation_provenance"] = dict(original_samples_unchanged=True,
        new_records=len(new), new_valid=sum(valid(s) for s in new),
        new_provider="Online Qwen3.7-Flash; reference and modification only",
        failed_records="Retained, explicit DQU fallback", validation="Unchanged original online-Qwen file")
    return result


def process_matches(pid, output):
    try:
        path = Path(f"/proc/{pid}")
        if path.joinpath("stat").read_text().split(") ", 1)[1].split()[0] == "Z":
            return False
        command = path.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode()
    except FileNotFoundError:
        return False
    if "generate_fiq_augmentation_online.py" not in command or str(output) not in command:
        raise RuntimeError("Adopted generation PID identity changed")
    if Path(os.readlink(path / "cwd")).resolve() != ROOT:
        raise RuntimeError("Generation working directory mismatch")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "overlay", "output", "clip-checkpoint", "dqu-checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--generation-pid", type=int, required=True)
    parser.add_argument("--generation-start-unix", type=float, required=True)
    args = parser.parse_args()
    import fcntl
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = args.output / "ablation_status.json"
    if status_path.exists():
        raise FileExistsError("Inspect existing experiment before any explicit resume")
    state = dict(status="preparing", started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 runs=[], maximum_wall_seconds=10800, generation_maximum_seconds=5400)

    def save(**values):
        state.update(values)
        state["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(status_path, state)

    try:
        require_space(args.output)
        manifest = json.loads((args.overlay / "augmentation_manifest.json").read_text())
        _, accepted = validate(args.root, manifest)
        original = json.loads((args.root / "captions/cap.dress.train.json").read_text())
        expanded = json.loads((args.overlay / "captions/cap.dress.train.json").read_text())
        if len(original) != 5985 or len(accepted) != 101 or len(expanded) != 6086 or expanded[:5985] != original:
            raise ValueError("Locked dataset sizes/prefix mismatch")
        for name in ("cap.dress.val.json", "cap.dress.test.json"):
            if file_sha(args.root / "captions" / name) != file_sha(args.overlay / "captions" / name):
                raise ValueError("Held-out annotations changed")
        budget = len(expanded)
        base_train = args.root / "captions/structured_edits_dress_train_qwen37flash.json"
        val = args.root / "captions/structured_edits_dress_val_qwen37flash.json"
        protocol = dict(category="dress", seed=42, epochs=6, warmup_gate_epochs=2, batch_size=16,
            learning_rate=5e-5, hard_negative_weight=0, multi_positive="off", frozen_dqu=True,
            original_unique_queries=len(original), augmented_unique_queries=budget,
            examples_per_epoch_per_arm=budget, original_repeats_per_epoch=101,
            optimizer_updates_per_epoch_per_arm=(budget+15)//16,
            model_selection="R@10+R@50 on unchanged full validation", eval_queries=2017, gallery_images=3817,
            eval_limit=0, training_truncation=False, added_images=0, online_api_budget_cny=1, online_max_calls=202,
            preparation_wall_start=args.generation_start_unix, maximum_wall_seconds=10800,
            source_hashes={str(p):file_sha(p) for p in [args.overlay/'augmentation_manifest.json', base_train,val,args.dqu_checkpoint]},
            code_hashes={name:file_sha(ROOT/name) for name in ['src/newTrain.py','src/newModel.py','src/newDataset.py','src/generate_structured_edits.py','src/generate_fiq_augmentation_online.py','src/epoch_budget_sampler.py', 'scripts/run_fiq_augmentation_ablation.py']},
            interpretation="Single-seed pilot, full augmented Dress training set; not a three-category FashionIQ mean")
        write_json(args.output / "protocol.json", protocol)
        save(status="waiting_for_new_structures", protocol=protocol)
        extra_path = args.output / "structured_new101_online.json"
        while process_matches(args.generation_pid, extra_path):
            require_space(args.output)
            if time.time()-args.generation_start_unix > 5400:
                os.kill(args.generation_pid, signal.SIGTERM)
                raise TimeoutError("90-minute structure preparation budget reached; partial responses retained")
            time.sleep(20)
        extra = json.loads(extra_path.read_text())
        if (extra.get('model') != 'qwen3.7-flash' or extra.get('state') != 'complete'
                or extra.get('completed_queries') != 101 or extra.get('cost',{}).get('accounted_upper_cny',2)>1):
            raise ValueError('Online generation incomplete, wrong provider or over budget')
        merged = merge_structured(json.loads(base_train.read_text()), extra,
                                  [(r['candidate'],r['target']) for r in accepted])
        merged_path = args.output / "structured_train_merged.json"
        write_json(merged_path, merged)
        save(status="structures_ready", new_valid=merged['augmentation_provenance']['new_valid'],
             merged_sha256=file_sha(merged_path), validation_sha256=file_sha(val), online_cost=extra['cost'])
        # All generator processes must release the GPU before either paired arm.
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError("GPU is occupied after structure generation; no training launched")
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        for arm, root in [("original_padded",args.root),("augmented101",args.overlay)]:
            require_space(args.output)
            remaining = 10800-(time.time()-args.generation_start_unix)
            if remaining < 600:
                raise TimeoutError("Insufficient remaining paired-trial budget")
            command = [sys.executable,"-u","newTrain.py","--dataset","dress","--fashioniq-split","original-split",
                "--fashioniq-path",str(root),"--clip-checkpoint",str(args.clip_checkpoint),"--dqu-checkpoint",str(args.dqu_checkpoint),
                "--structured-train-path",str(merged_path),"--structured-val-path",str(val),"--multi-positive","off",
                "--seed","42","--num-epochs","6","--batch-size","16","--num-workers","4","--patience","20",
                "--hard-negative-weight","0","--train-samples-per-epoch",str(budget),"--run-name",arm,
                "--model-dir",str(args.output/'checkpoints')]
            row=dict(arm=arm,status="running",command=command);state['runs'].append(row);save(status="training_"+arm)
            with (args.output/(arm+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT/'src',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                row['pid']=child.pid;save()
                try:
                    code=child.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise TimeoutError("Three-hour paired-trial budget reached")
            row.update(returncode=code,status='complete' if code==0 else 'failed');save()
            if code:
                raise RuntimeError(f"{arm} failed; preserve log and inspect")
            metric_path=args.output/'checkpoints'/f'dress_{arm}_seed42_best.json'
            row['best_metrics']=json.loads(metric_path.read_text());save()
        before,after=[r['best_metrics'] for r in state['runs']]
        if file_sha(args.dqu_checkpoint) != protocol['source_hashes'][str(args.dqu_checkpoint)]:
            raise RuntimeError("Protected DQU checkpoint changed")
        delta={key:after[key]-before[key] for key in ['dress_r1','dress_r10','dress_r50']}
        save(status="complete",delta_percentage_points=delta,
             interpretation="Final single-seed data ablation; no significance claim. Shirt/Toptee not retrained.")
    except BaseException as error:
        save(status="failed",error_type=type(error).__name__,error=str(error))
        raise


if __name__ == "__main__":
    main()
