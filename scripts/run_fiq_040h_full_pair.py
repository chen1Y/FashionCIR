"""Run full FashionIQ 040E Flash and the native per-category DQU-CIR baseline."""

import argparse
from collections import Counter
import csv
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

from run_fiq_040e_flash import read, sha
from run_fiq_context_generation_041 import write


COUNTS = {"dress": {"train": 5639, "val": 2017}, "shirt": {"train": 5312, "val": 2038},
          "toptee": {"train": 5291, "val": 1961}}
NATIVE = {"dress": (5985, 2017, 3817), "shirt": (5988, 2038, 6346), "toptee": (6027, 1961, 5373)}


def prepare(root, source, data):
    generation = read(source / "state.json")
    template = read(source / "training_plan_template.json")
    if generation["status"] not in ("complete", "complete_with_unknown"):
        raise ValueError("The Flash export is not complete")
    with sqlite3.connect((source / "responses.sqlite3").as_uri() + "?mode=ro", uri=True, timeout=10) as db:
        requests = dict(db.execute("SELECT status, COUNT(*) FROM requests GROUP BY status"))
    if requests != generation["requests"] or set(requests) - {"returned", "unknown"}:
        raise ValueError("The Flash request ledger still has unfinished work")
    for name, digest in template["source_sha256"].items():
        if sha(root / "code" / name) != digest:
            raise ValueError(f"Original full Flash trainer source differs: {name}")
    inventory_file = root / "texts" / "inventory.json"
    inventory = read(inventory_file)
    if inventory["rows"] != 24016 or inventory["totals"] != generation["text_counts"]:
        raise ValueError("Flash export does not cover all original triplets")
    usage, native_sources = {}, {}
    for category, counts in COUNTS.items():
        meta_path = root / "metadata" / f"{category}.json"
        source_spec = template["inputs"][category]
        if sha(meta_path) != source_spec["metadata_sha256"]:
            raise ValueError(f"Full training metadata changed for {category}")
        meta = read(meta_path)
        usage[category] = {}
        native_sources[category] = {}
        for split, expected in counts.items():
            spec = inventory["splits"][f"{category}_{split}"]
            path = root / "texts" / spec["path"]
            if sha(path) != spec["sha256"]:
                raise ValueError(f"Flash export digest differs for {category}/{split}")
            with path.open(encoding="utf-8") as handle:
                exported = list(map(json.loads, handle))
            annotation_file = data / "captions" / f"cap.{category}.{split}.json"
            caption_file = data / "captions" / f"image_captions_{category}_{split}.json"
            raw, captions = read(annotation_file), read(caption_file)
            native_count = NATIVE[category][0 if split == "train" else 1]
            if len(exported) != len(raw) or len(raw) != spec["rows"] or len(raw) != native_count:
                raise ValueError(f"Raw FashionIQ denominator changed for {category}/{split}")
            if any(row["candidate"] not in captions for row in raw):
                raise ValueError(f"Native BLIP-2 caption missing for {category}/{split}")
            native_sources[category].update({str(p): sha(p) for p in (annotation_file, caption_file)})
            records = meta["records"] if split == "train" else [
                dict(row, source_index=i) for i, row in enumerate(raw)
            ]
            if len(records) != expected:
                raise ValueError(f"Matched denominator differs for {category}/{split}")
            tally = Counter()
            for record in records:
                index = record["source_index"]
                row, original = exported[index], raw[index]
                if ((row["category"], row["split"], row["source_index"]) != (category, split, index)
                        or any(row[k] != original[k] or row[k] != record[k] for k in ("candidate", "target"))
                        or row["fragments"] != original["captions"]):
                    raise ValueError(f"Query identity differs for {category}/{split}/{index}")
                if split == "train" and meta["image_sha256"][row["candidate"]] != row["reference_sha256"]:
                    raise ValueError(f"Reference JPEG digest differs for {category}/{split}/{index}")
                tally[row["status"]] += 1
            usage[category][split] = dict(tally)
        keyword_file = data / "captions" / f"keywords_in_mods_{category}.json"
        correction_file = data / "captions" / f"correction_dict_{category}.json"
        gallery_file = data / "image_splits" / f"split.{category}.val.json"
        keywords, gallery = read(keyword_file), set(read(gallery_file))
        if len(gallery) != NATIVE[category][2]:
            raise ValueError(f"Native gallery denominator differs for {category}")
        for split in ("train", "val"):
            for row in read(data / "captions" / f"cap.{category}.{split}.json"):
                ref, target = row["candidate"], row["target"]
                if f"{ref}_{target}" not in keywords or (split == "val" and {ref, target} - gallery):
                    raise ValueError(f"Native DQU keyword or validation ID missing for {category}/{split}")
        native_sources[category].update({str(p): sha(p) for p in (keyword_file, correction_file, gallery_file)})
    flash = root / "flash"
    flash.mkdir()
    for name in ("code", "texts", "metadata", "weights"):
        (flash / name).symlink_to(root / name, target_is_directory=True)
    template.update(experiment="040H full 040E Flash", control="040H native DQU-CIR",
                    text_inventory_sha256=sha(inventory_file))
    template["full_caption"].update(generation_plan_sha256=sha(source / "plan.json"),
                                     generation_status=generation["status"],
                                     unknown_fallback=generation["text_counts"].get("unknown_fallback", 0))
    write(flash / "plan.json", template)
    write(root / "preflight.json", dict(flash_samples=COUNTS, native_samples=NATIVE, native_sources=native_sources,
                                        native_code={p.name: sha(p) for p in (root / "dqu_code").glob("*.py")},
                                        text_inventory_sha256=sha(inventory_file), text_used=usage,
                                        generation=generation, api_calls=0))
    print(json.dumps(dict(flash_samples=COUNTS, native_samples=NATIVE, text_used=usage), ensure_ascii=False))


def snapshot():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu",
                                      "--format=csv,noheader,nounits"], text=True)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                         "--format=csv,noheader,nounits"], text=True)
    by_gpu = {}
    for gpu, pid, memory in csv.reader(processes.splitlines(), skipinitialspace=True):
        by_gpu.setdefault(gpu, {})[pid] = int(memory)
    return {int(i): (uuid, int(free), int(use), by_gpu.get(uuid, {}))
            for i, uuid, free, use in csv.reader(output.splitlines(), skipinitialspace=True)}


def choose_gpu(arm, root, state, completed):
    required = 19000 if arm == "flash" else 19456
    while not (root / "cancel.request").exists():
        first = snapshot()
        time.sleep(5)
        second = snapshot()
        for i in sorted(second, key=lambda index: (bool(second[index][3]), index != 1, index)):
            uuid, free, use, procs = second[i]
            old_uuid, old_free, old_use, old_procs = first[i]
            if (uuid != old_uuid or min(free, old_free) < required or max(use, old_use) > 75
                    or set(procs) != set(old_procs)
                    or any(abs(procs[p] - old_procs[p]) > 256 for p in procs)):
                continue
            return i, uuid
        write(state, dict(status="waiting_gpu", arm=arm, completed=completed, minimum_free_mib=required))
        time.sleep(30)
    return None


def run(root, data):
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = root / "queue_state.json"
        if state.exists():
            raise FileExistsError("Full matched queue already started")
        preflight = read(root / "preflight.json")
        if sha(root / "texts" / "inventory.json") != preflight["text_inventory_sha256"]:
            raise ValueError("Flash text manifest changed")
        for name, digest in preflight["native_code"].items():
            if sha(root / "dqu_code" / name) != digest:
                raise ValueError(f"Native DQU source changed: {name}")
        flash = root / "flash"
        dqu = root / "dqu"
        dqu.mkdir()
        completed = {}
        stages = [("flash", flash), *((f"dqu_{cat}", dqu / cat) for cat in NATIVE)]
        for arm, directory in stages:
            card = choose_gpu(arm, root, state, completed)
            if card is None:
                write(state, dict(status="cancelled", completed=completed))
                return
            gpu_index, gpu = card
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                       PYTHONPATH=f"{root / 'code'}:/home/lcy/fashionCIR/src")
            if arm == "flash":
                plan = read(flash / "plan.json")
                plan.update(gpu=gpu, gpu_index=gpu_index)
                write(flash / "plan.json", plan)
                args = [str(root / "code" / "run_fiq_vit_lora_039a_earlystop.py"),
                        "--root", str(flash), "--gpu", gpu]
            else:
                category = arm.removeprefix("dqu_")
                for path, digest in preflight["native_sources"][category].items():
                    if sha(Path(path)) != digest:
                        raise ValueError(f"Native DQU data changed: {category}")
                directory.mkdir()
                args = [str(root / "dqu_code" / "dquTrain.py"), "--dataset", category,
                        "--fashioniq-split", "original-split", "--fashioniq-path", str(data),
                        "--clip-checkpoint", str(root / "weights" / "open_clip_pytorch_model.bin"),
                        "--seed", "42", "--batch-size", "16", "--num-workers", "4",
                        "--eval-every", "1", "--patience", "3", "--run-name", "040h_native",
                        "--model-dir", str(directory / "checkpoints"),
                        "--validation-metrics-csv", str(directory / "validation_metrics.csv"),
                        "--gradient-checkpointing", "--adamw-no-foreach", "--memory-fraction", ".74"]
            write(state, dict(status="running", arm=arm, gpu=gpu_index, completed=completed))
            with (directory / "train.log").open("wb") as log:
                result = subprocess.run(["nice", "-n", "8", sys.executable, "-u", *args],
                                        cwd="/home/lcy/fashionCIR", env=env, stdout=log, stderr=log)
            completed[arm] = result.returncode
        write(state, dict(status="failed" if any(completed.values()) else "complete", completed=completed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-from", type=Path)
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare_from:
        prepare(args.root, args.prepare_from, args.data)
    else:
        run(args.root, args.data)
