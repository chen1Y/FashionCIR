"""Freeze a larger Dress subset of settled Flash responses and train two matched arms."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

from run_fiq_040e_flash import gpu_snapshot, read, sha
from run_fiq_context_generation_041 import write


CATEGORY = "dress"
COUNTS = {"train": 1024, "val": 256}
GPU_INDEX = 1


def select_text(row, tokenizer):
    response = json.loads(row["response"])
    if response.get("model") != "qwen3.8-flash" or "error" in response:
        raise ValueError("A settled sample has no verified Flash response")
    answer = response["answer"]
    fallback = json.loads(row["fallback"])
    if response["finish"] != "stop" or not response["thinking_present"] or response["flags"] or not isinstance(answer, dict):
        return "generation_fallback", fallback
    drop = ", ".join(answer["drop_attrs"])
    main = answer["final_text"]
    if len(tokenizer.encode(main)) > 75 or len(tokenizer.encode(drop)) > 75:
        return "token_budget_fallback", fallback
    return "qwen_used", dict(main_context_text=main, kept_context=answer["Tkep"], drop_attrs=drop, origin="qwen")


def prepare(root, source, pilot):
    from open_clip import get_tokenizer

    generation = read(source / "plan.json")
    pilot_plan = read(pilot / "plan.json")
    if generation["model"] != "qwen3.8-flash" or generation["prompt_sha256"] != pilot_plan["pilot"]["prompt_sha256"]:
        raise ValueError("Expected the same Flash model and prompt as the pilot")
    pilot_cases = pilot / "cases.json"
    if sha(pilot_cases) != pilot_plan["pilot"]["cases_sha256"]:
        raise ValueError("Pilot case manifest changed")
    old = read(pilot_cases)
    full = read(source / "training_plan_template.json")
    spec = full["inputs"][CATEGORY]
    meta = root / "metadata" / f"{CATEGORY}.json"
    if sha(meta) != spec["metadata_sha256"]:
        raise ValueError("Training metadata does not match the full experiment")
    train_records = read(meta)["records"]
    offsets = {row["source_index"]: i for i, row in enumerate(train_records)}
    with sqlite3.connect((source / "responses.sqlite3").as_uri() + "?mode=ro", uri=True, timeout=15) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("""
            SELECT s.*, r.reference_sha256, r.fragments, r.response, r.origin
            FROM samples s JOIN requests r ON s.request_id=r.id
            WHERE s.category=? AND r.status='returned'
        """, (CATEGORY,)).fetchall()
    tokenizer = get_tokenizer("ViT-H-14")
    texts, manifest, cases, selected_offsets = {}, {}, [], {}
    for split, count in COUNTS.items():
        existing = {r["source_index"] for r in old if (r["category"], r["split"]) == (CATEGORY, split)}
        eligible = {r["source_index"]: r for r in rows
                    if r["split"] == split and (split == "val" or r["source_index"] in offsets)}
        if len(eligible) < count or not existing <= eligible.keys():
            raise ValueError(f"Only {len(eligible)} eligible settled Dress {split} responses; need {count}")
        extras = sorted(eligible.keys() - existing,
                        key=lambda i: hashlib.sha256(f"42:dress:{split}:{i}".encode()).digest())
        ids = sorted(existing | set(extras[:count-len(existing)]),
                     key=offsets.__getitem__ if split == "train" else None)
        selected_offsets[split] = [offsets[i] if split == "train" else i for i in ids]
        text_rows = []
        for index, offset in zip(ids, selected_offsets[split]):
            row = eligible[index]
            status, text = select_text(row, tokenizer)
            identity = dict(category=CATEGORY, split=split, source_index=index, candidate=row["candidate"],
                            reference_sha256=row["reference_sha256"])
            cases.append(dict(**identity, offset=offset, fragments=json.loads(row["fragments"])))
            text_rows.append(dict(**identity, target=row["target"], status=status, **text))
        texts[split] = text_rows
    text_dir = root / "texts"
    text_dir.mkdir()
    for split, rows in texts.items():
        path = text_dir / f"{CATEGORY}_{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest[f"{CATEGORY}_{split}"] = dict(path=path.name, rows=len(rows), sha256=sha(path))
    write(text_dir / "inventory.json", dict(splits=manifest, rows=sum(COUNTS.values())))
    write(root / "cases.json", cases)
    pair = dict(experiment="040G matched Dress DQU-CIR", categories=[CATEGORY],
                inputs={CATEGORY: spec}, comparison_queries={CATEGORY: COUNTS},
                pilot=dict(cases_sha256=sha(root / "cases.json"), offsets={CATEGORY: selected_offsets}))
    write(root / "plan.json", pair)
    flash = root / "flash"
    flash.mkdir()
    for name in ("code", "texts", "metadata", "weights"):
        (flash / name).symlink_to(root / name, target_is_directory=True)
    full.update(experiment="040G matched Dress 040E Flash", categories=[CATEGORY], inputs={CATEGORY: spec},
                source_sha256={p.name: sha(p) for p in (root / "code").iterdir() if p.is_file()},
                text_inventory_sha256=sha(text_dir / "inventory.json"))
    full["full_caption"].update(offsets={CATEGORY: selected_offsets}, generated_rows=sum(COUNTS.values()),
                                train={CATEGORY: COUNTS["train"]}, validation={CATEGORY: COUNTS["val"]})
    full["protocol"].update(training_queries=COUNTS["train"], updates_per_epoch=64)
    full["protocol"]["validation"]["queries"] = COUNTS["val"]
    write(flash / "plan.json", full)
    write(root / "preflight.json", dict(rows=COUNTS, cases_sha256=sha(root / "cases.json"),
                                        source_plan_sha256=sha(source / "plan.json"),
                                        text_inventory_sha256=sha(text_dir / "inventory.json"), api_calls=0,
                                        sampling="retain original pilot and add hash-ordered settled responses"))
    print(json.dumps(read(root / "preflight.json"), ensure_ascii=False))


def run(root, data):
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = root / "queue_state.json"
        if state.exists():
            raise FileExistsError("Queue already started")
        preflight = read(root / "preflight.json")
        if sha(root / "cases.json") != preflight["cases_sha256"]:
            raise ValueError("Matched samples changed before launch")
        first = gpu_snapshot()
        time.sleep(5)
        second = gpu_snapshot()
        if GPU_INDEX not in first or first[GPU_INDEX] != second.get(GPU_INDEX):
            write(state, dict(status="waiting_gpu", gpu=GPU_INDEX))
            return
        gpu = second[GPU_INDEX]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                   PYTHONPATH=f"{root / 'code'}:/home/lcy/fashionCIR/src")
        flash = root / "flash"
        plan = read(flash / "plan.json")
        plan.update(gpu=gpu, gpu_index=GPU_INDEX)
        write(flash / "plan.json", plan)
        dqu = root / "dqu"
        dqu.mkdir()
        cmds = {
            "flash": [str(root / "code" / "run_fiq_vit_lora_039a_earlystop.py"), "--root", str(flash), "--gpu", gpu],
            "dqu": [str(root / "dqu_code" / "run_fiq_dqu_matched_040f.py"), "--root", str(dqu),
                    "--control", str(root), "--data", str(data), "--gpu", gpu,
                    "--checkpoint", str(root / "weights" / "open_clip_pytorch_model.bin")],
        }
        arms = {}
        for name, args in cmds.items():
            if gpu_snapshot().get(GPU_INDEX) != gpu:
                write(state, dict(status="waiting_gpu", gpu=GPU_INDEX, completed=arms))
                return
            write(state, dict(status="running", gpu=GPU_INDEX, arm=name, completed=arms))
            with (root / name / "train.log").open("wb") as log:
                result = subprocess.run(["nice", "-n", "8", sys.executable, "-u", *args],
                                        cwd="/home/lcy/fashionCIR", stdout=log, stderr=log, env=env)
            arms[name] = result.returncode
        write(state, dict(status="failed" if any(arms.values()) else "complete", gpu=GPU_INDEX, completed=arms))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-from", type=Path)
    action.add_argument("--run", action="store_true")
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    if args.prepare_from:
        prepare(args.root, args.prepare_from, args.pilot)
    else:
        run(args.root, args.data)
