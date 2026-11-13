"""Extend the 040E text inputs to all labeled FashionIQ train/val triplets."""

import argparse
import base64
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import time

from build_040_conflict_free_text import prepare
from fiq_text_branch_math_027 import corrected_modification
from run_fiq_context_generation_041 import BASE_URL, SETTINGS, call, checked_endpoint, write


MODEL = "qwen3.7-plus"
COUNTS = {"dress": (5985, 2017), "shirt": (5988, 2038), "toptee": (6027, 1961)}
PROMPT_NAME = "fiq_context_generation_041_prompt.txt"
INPUT_CNY = 2
OUTPUT_CNY = 8
SCHEMA = """
CREATE TABLE requests (
    id TEXT PRIMARY KEY, candidate TEXT NOT NULL, reference_sha256 TEXT NOT NULL,
    fragments TEXT NOT NULL, origin TEXT NOT NULL, status TEXT NOT NULL, response TEXT,
    attempts INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE samples (
    category TEXT NOT NULL, split TEXT NOT NULL, source_index INTEGER NOT NULL,
    target TEXT NOT NULL, candidate TEXT NOT NULL,
    request_id TEXT NOT NULL REFERENCES requests(id), fallback TEXT NOT NULL,
    PRIMARY KEY(category, split, source_index)
);
CREATE INDEX requests_by_status ON requests(status);
"""


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def identity(row):
    return row["category"], row.get("split", "val"), row["source_index"]


def fingerprint(prompt_sha, jpeg_sha, fragments):
    return hashlib.sha256(packed([MODEL, SETTINGS, prompt_sha, jpeg_sha, fragments]).encode()).hexdigest()


def connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    return db


def load_previous(root, control, prompt_sha):
    plan = read(control / "plan.json")
    pilot = plan["pilot"]
    gen_file = control / "generation.json"
    cases_file = control / "cases.json"
    if sha(gen_file) != pilot["generation_sha256"] or sha(cases_file) != pilot["cases_sha256"]:
        raise ValueError("040E pilot inputs do not match the training plan")
    generation = read(gen_file)
    cases = read(cases_file)
    if (generation["status"] != "complete" or generation["model"] != MODEL
            or generation["settings"] != SETTINGS or generation["prompt_sha256"] != prompt_sha
            or generation["target_images_sent"] != 0 or len(cases) != 576
            or len(generation["answers"]) != len(cases)):
        raise ValueError("040E pilot generation protocol differs")
    history = {identity(c): (c, answer, "040C_pilot") for c, answer in zip(cases, generation["answers"])}
    if len(history) != len(cases):
        raise ValueError("Duplicate 040E pilot identities")
    paired = read(root / "paired_041.json")
    if paired["status"] != "complete" or paired["settings"] != SETTINGS or paired["prompt_sha256"] != prompt_sha:
        raise ValueError("041 paired prompt or generation settings differ")
    if paired["target_images_sent"] != 0:
        raise ValueError("041 input included a target image")
    for pair in paired["pairs"]:
        case = pair["case"]
        answer, = [o for o in pair["models"] if o["model"] == MODEL]
        history.setdefault(identity(case), (case, answer, "041_paired"))
    for _, answer, _ in history.values():
        if answer["model"] != MODEL or "error" in answer:
            raise ValueError("A historical generation response is missing or uses a different model")
    return history


def fallback_rows(control, sources):
    inventory_file = control / "texts" / "inventory.json"
    if sha(inventory_file) != read(control / "plan.json")["text_inventory_sha256"]:
        raise ValueError("040E fallback inventory differs")
    sources[str(inventory_file)] = sha(inventory_file)
    for name, spec in read(inventory_file)["splits"].items():
        category, split = name.split("_")
        path = inventory_file.parent / spec["path"]
        if sha(path) != spec["sha256"]:
            raise ValueError(f"Fallback file changed for {name}")
        sources[str(path)] = spec["sha256"]
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                yield (category, split, row["source_index"]), row


def setup(root, data, control, workers, rpm):
    from open_clip import get_tokenizer

    prompt_sha = sha(root / "code" / PROMPT_NAME)
    history = load_previous(root, control, prompt_sha)
    paths = (control / "cases.json", control / "generation.json", control / "plan.json", root / "paired_041.json")
    sources = {str(path): sha(path) for path in paths}
    fallbacks = dict(fallback_rows(control, sources))
    tokenizer = get_tokenizer("ViT-H-14")
    canonical, requests, samples = {}, {}, []
    image_hashes = {}
    used_history, used_fallback = set(), set()
    for category, counts in COUNTS.items():
        for split, count in zip(("train", "val"), counts):
            annotation_file = data / "captions" / f"cap.{category}.{split}.json"
            captions_file = data / "captions" / f"image_captions_{category}_{split}.json"
            correction_file = data / "captions" / f"correction_dict_{category}.json"
            for path in (annotation_file, captions_file, correction_file):
                sources[str(path)] = sha(path)
            annotations = read(annotation_file)
            ref_captions, corrections = read(captions_file), read(correction_file)
            if len(annotations) != count:
                raise ValueError(f"FashionIQ count differs for {category}/{split}")
            for index, row in enumerate(annotations):
                ref, target, fragments = row["candidate"], row["target"], row["captions"]
                if (not isinstance(fragments, list) or len(fragments) != 2
                        or not all(isinstance(s, str) for s in fragments)
                        or not all(re.fullmatch(r"[A-Z0-9]{10}", v) for v in (ref, target))):
                    raise ValueError(f"Malformed FashionIQ triplet {category}/{split}/{index}")
                if ref not in image_hashes:
                    jpeg = (data / "resized_image" / f"{ref}.jpg").read_bytes()
                    if not jpeg.startswith(b"\xff\xd8"):
                        raise ValueError(f"Invalid reference JPEG {ref}")
                    image_hashes[ref] = hashlib.sha256(jpeg).hexdigest()
                jpeg_sha = image_hashes[ref]
                key = (category, split, index)
                request_key = fingerprint(prompt_sha, jpeg_sha, fragments)
                if key in history:
                    case, answer, origin = history[key]
                    if (case["candidate"], case["reference_sha256"], case["fragments"]) != (ref, jpeg_sha, fragments):
                        raise ValueError(f"Saved reference or edit mismatch at {key}")
                    request_id = f"{origin}:{category}:{split}:{index}"
                    canonical.setdefault(request_key, request_id)
                    requests[request_id] = (ref, jpeg_sha, packed(fragments), origin, "returned", packed(answer))
                    used_history.add(key)
                else:
                    request_id = request_key
                    requests.setdefault(request_key, (ref, jpeg_sha, packed(fragments), "new", "pending", None))
                if key in fallbacks:
                    fallback = fallbacks[key]
                    if fallback["candidate"] != ref:
                        raise ValueError(f"040A fallback reference changed at {key}")
                    fallback_origin = "040A"
                    used_fallback.add(key)
                else:
                    fallback = prepare(
                        dict(category=category, source_index=index, candidate=ref, drop_attrs="", target_attrs={}),
                        ref_captions.get(ref, ""), corrected_modification(fragments, corrections), tokenizer,
                    )
                    fallback_origin = "040A_text_rules_no_prior_drop"
                fallback = {field: fallback[field] for field in ("main_context_text", "kept_context", "drop_attrs")}
                samples.append((*key, target, ref, request_id, packed(dict(**fallback, origin=fallback_origin))))
    if used_history != set(history) or used_fallback != set(fallbacks):
        raise ValueError("Historical generation or fallback includes unmatched rows")

    # Bind identical prompts to one response, preserving the original 040E pilot results by row.
    samples = [(*row[:5], canonical.get(row[5], row[5]), row[6]) for row in samples]
    referenced = {row[5] for row in samples}
    requests = {key: value for key, value in requests.items() if key in referenced}
    pending = sum(values[4] == "pending" for values in requests.values())
    usage = [answer["usage"] for _, answer, _ in history.values() if answer.get("usage")]
    per_request_cny = sum(u["prompt_tokens"] * INPUT_CNY + u["completion_tokens"] * OUTPUT_CNY
                          for u in usage) / len(usage) / 1e6
    plan = dict(model=MODEL, settings=SETTINGS, prompt_sha256=prompt_sha, data=str(data),
                counts={c: dict(train=n, val=m) for c, (n, m) in COUNTS.items()},
                total_rows=len(samples), historical_rows=len(history), unique_requests=len(requests),
                new_requests=pending, reference_images=len(image_hashes), target_images_sent=0,
                planned_list_price_cny=round(pending * per_request_cny, 2), workers=workers, rpm=rpm,
                endpoint=checked_endpoint(BASE_URL), sources=sources,
                source_sha256={p.name: sha(p) for p in (root / "code").glob("*") if p.is_file()})
    path = root / "responses.sqlite3"
    if path.exists() or (root / "plan.json").exists():
        raise FileExistsError("The full text run already has a plan or response ledger")
    db_tmp = path.with_suffix(".building")
    if db_tmp.exists():
        raise FileExistsError(db_tmp)
    with closing(connect(db_tmp)) as db, db:
        db.executescript(SCHEMA)
        db.executemany("INSERT INTO requests VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                       ((key, *values) for key, values in requests.items()))
        db.executemany("INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?)", samples)
    db_tmp.replace(path)
    write(root / "plan.json", plan)
    print(json.dumps({k: plan[k] for k in ("counts", "total_rows", "historical_rows", "unique_requests",
                                          "new_requests", "planned_list_price_cny", "workers", "rpm")}))


def progress(root, db, status, **extra):
    counts = dict(db.execute("SELECT status, COUNT(*) FROM requests GROUP BY status").fetchall())
    done, = db.execute("""SELECT COUNT(*) FROM samples JOIN requests ON requests.id=samples.request_id
                         WHERE requests.status='returned'""").fetchone()
    write(root / "state.json", dict(status=status, pid=os.getpid(), rows_with_response=done,
                                   requests=counts, **extra))


def send(row, plan, key, prompt):
    jpeg = (Path(plan["data"]) / "resized_image" / f"{row['candidate']}.jpg").read_bytes()
    if hashlib.sha256(jpeg).hexdigest() != row["reference_sha256"]:
        raise ValueError(f"Reference changed for {row['candidate']}")
    url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    return call(plan["endpoint"], key, MODEL, url, prompt, json.loads(row["fragments"]))


def export(root, db):
    from open_clip import get_tokenizer

    tokenizer = get_tokenizer("ViT-H-14")
    output = root / "texts"
    output.mkdir(exist_ok=True)
    manifest, totals = {}, Counter()
    for category, counts in COUNTS.items():
        for split, count in zip(("train", "val"), counts):
            path = output / f"{category}_{split}.jsonl"
            tmp = path.with_suffix(".tmp")
            counts_by_status = Counter()
            rows = db.execute("""SELECT s.*, r.reference_sha256, r.fragments, r.status, r.response, r.origin
                                 FROM samples s JOIN requests r ON s.request_id=r.id
                                 WHERE s.category=? AND s.split=? ORDER BY s.source_index""", (category, split))
            with tmp.open("w", encoding="utf-8") as file:
                for index, row in enumerate(rows):
                    if row["source_index"] != index:
                        raise ValueError(f"Unexpected row order in {path.name}")
                    response = json.loads(row["response"]) if row["response"] else {}
                    answer = response.get("answer")
                    fallback = json.loads(row["fallback"])
                    status = "generation_fallback"
                    if row["status"] != "returned":
                        status = f"{row['status']}_fallback"
                    elif (isinstance(answer, dict) and response["finish"] == "stop"
                          and response["thinking_present"] and not response["flags"]):
                        drop = ", ".join(answer["drop_attrs"])
                        main = answer["final_text"]
                        if len(tokenizer.encode(main)) <= 75 and len(tokenizer.encode(drop)) <= 75:
                            fallback = dict(main_context_text=main, kept_context=answer["Tkep"],
                                            drop_attrs=drop, origin="qwen")
                            status = "qwen_used"
                        else:
                            status = "token_budget_fallback"
                    counts_by_status[status] += 1
                    file.write(packed(dict(
                        category=category, split=split, source_index=index, candidate=row["candidate"],
                        target=row["target"], fragments=json.loads(row["fragments"]),
                        reference_sha256=row["reference_sha256"], request_id=row["request_id"],
                        response_origin=row["origin"], status=status, **fallback,
                    )) + "\n")
            if sum(counts_by_status.values()) != count:
                raise ValueError(f"Incorrect export size for {path.name}")
            tmp.replace(path)
            totals.update(counts_by_status)
            manifest[f"{category}_{split}"] = dict(path=path.name, rows=count, sha256=sha(path),
                                                   status_counts=dict(counts_by_status))
    write(output / "inventory.json", dict(splits=manifest, totals=dict(totals), rows=sum(totals.values())))
    return dict(totals)


def run(root, plan, key, retry_rejected):
    prompt = (root / "code" / PROMPT_NAME).read_text(encoding="utf-8")
    reason = None
    with closing(connect(root / "responses.sqlite3")) as db:
        with db:
            db.execute("UPDATE requests SET status='unknown' WHERE status='in_flight'")
            if retry_rejected:
                db.execute("UPDATE requests SET status='pending' WHERE status='rejected'")
        progress(root, db, "running")
        pending = iter(db.execute("SELECT * FROM requests WHERE status='pending' ORDER BY rowid").fetchall())
        active = {}
        next_send, last_report = 0., 0.
        exhausted = False
        verified_access = False
        consecutive_errors = 0
        with ThreadPoolExecutor(max_workers=plan["workers"]) as pool:
            while active or (not exhausted and not reason):
                if (root / "cancel.request").exists():
                    reason = "cancelled"
                now = time.monotonic()
                # Require one real response before filling the concurrent queue.
                capacity = plan["workers"] if verified_access else 1
                if not reason and not exhausted and len(active) < capacity and now >= next_send:
                    row = next(pending, None)
                    if row is None:
                        exhausted = True
                    else:
                        with db:
                            db.execute("UPDATE requests SET status='in_flight', attempts=attempts+1 WHERE id=?", (row["id"],))
                        active[pool.submit(send, row, plan, key, prompt)] = row
                        next_send = now + 60 / plan["rpm"]
                done, _ = wait(active, timeout=.15, return_when=FIRST_COMPLETED) if active else ((), ())
                for future in done:
                    row = active.pop(future)
                    try:
                        response = future.result()
                    except Exception as exc:
                        response = dict(error=type(exc).__name__)
                        reason = "reference_or_worker_error"
                    error = response.get("error")
                    status = "returned"
                    if error:
                        status = "rejected" if re.fullmatch(r"HTTP 4\d\d", error) and error != "HTTP 408" else "unknown"
                        consecutive_errors += 1
                        if status == "rejected" or not verified_access or consecutive_errors >= 4:
                            reason = "provider_error"
                    else:
                        verified_access = True
                        consecutive_errors = 0
                    with db:
                        db.execute("UPDATE requests SET status=?, response=? WHERE id=?",
                                   (status, packed(response), row["id"]))
                    if error:
                        print(f"{row['id']} {error} {response.get('provider_code', '')}", flush=True)
                if now - last_report >= 10:
                    progress(root, db, "running", draining_reason=reason)
                    last_report = now
                if not active and not exhausted and not reason and now < next_send:
                    time.sleep(min(next_send - now, .15))
        counts = dict(db.execute("SELECT status, COUNT(*) FROM requests GROUP BY status").fetchall())
        if reason or counts.get("pending") or counts.get("rejected"):
            progress(root, db, "paused", reason=reason)
            return
        progress(root, db, "exporting")
        totals = export(root, db)
        billed = Counter()
        for record, in db.execute("SELECT response FROM requests WHERE origin='new' AND status='returned'"):
            usage = json.loads(record).get("usage")
            if usage:
                billed.update({k: usage[k] for k in ("prompt_tokens", "completion_tokens") if k in usage})
            else:
                billed["missing_usage"] += 1
        progress(root, db, "complete_with_unknown" if counts.get("unknown") else "complete",
                 text_counts=totals, usage=dict(billed),
                 estimated_cny=(billed["prompt_tokens"] * INPUT_CNY + billed["completion_tokens"] * OUTPUT_CNY) / 1e6)


def launch(root, retry_rejected):
    import fcntl

    with (root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(root / "plan.json")
        for name, digest in plan["source_sha256"].items():
            if sha(root / "code" / name) != digest:
                raise ValueError(f"Prepared code differs: {name}")
        if plan["endpoint"] != checked_endpoint(BASE_URL) or plan["model"] != MODEL or plan["settings"] != SETTINGS:
            raise ValueError("Prepared API settings differ")
        for name, digest in plan["sources"].items():
            if sha(Path(name)) != digest:
                raise ValueError(f"Prepared input differs: {name}")
        state = root / "state.json"
        if state.exists() and read(state)["status"] in ("complete", "complete_with_unknown"):
            raise ValueError("The run already exported all rows")
        key = getpass.getpass("DashScope API key (hidden input): ").strip()
        if not key:
            raise ValueError("No API key was entered")
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        pid = os.fork()
        if pid:
            print(f"040E full text generation worker started, PID {pid}.", flush=True)
            return
        os.setsid()
        with open(os.devnull, "rb") as stdin, (root / "generation.log").open("ab", buffering=0) as log:
            os.dup2(stdin.fileno(), 0)
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            try:
                run(root, plan, key, retry_rejected)
            except Exception as exc:
                with closing(connect(root / "responses.sqlite3")) as db:
                    progress(root, db, "failed", reason=type(exc).__name__)
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_args = sub.add_parser("prepare")
    prepare_args.add_argument("--data", type=Path, required=True)
    prepare_args.add_argument("--control", type=Path, required=True)
    prepare_args.add_argument("--workers", type=int, default=64)
    prepare_args.add_argument("--rpm", type=int, default=150)
    launch_args = sub.add_parser("launch")
    launch_args.add_argument("--retry-rejected", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.workers < 1 or args.rpm < 1:
            parser.error("workers and rpm must be positive")
        setup(args.root, args.data, args.control, args.workers, args.rpm)
    else:
        launch(args.root, args.retry_rejected)


if __name__ == "__main__":
    main()
