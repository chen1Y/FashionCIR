"""Metered full FashionIQ generation using the existing Dress prompt and rules."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

from experiment_storage import require_space
from generate_shoes_online import Ledger, RateLimiter, BudgetStop
from shoes_online_pilot import Client, BASE_URL, MODEL, reference_data_url
from generate_structured_edits import (
    SYSTEM_PROMPT, build_user_prompt, load_fashioniq_queries, extract_json_object,
    validate_edit_program, process_semantics, make_output_sample, atomic_write_json,
)

CATEGORIES = ("shirt", "toptee")
EXPECTED = {"shirt": {"train": 5988, "val": 2038}, "toptee": {"train": 6027, "val": 1961}}
VERSION = "fashioniq-dress-prompt-full-20261022-v1"


def fingerprint(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def valid(sample):
    return (sample.get("status") == "ok" and sample.get("validation", {}).get("valid") is True
            and sample.get("semantic_validation", {}).get("valid") is True)


def check_complete(payload, rows, category, split):
    if (payload.get("state") != "complete" or payload.get("model") != MODEL
            or payload.get("category") != category or payload.get("split") != split
            or payload.get("query_sha256") != fingerprint(rows)
            or payload.get("protocol", {}).get("version") != VERSION):
        raise ValueError("Generation completion, provider or query fingerprint mismatch")
    samples = payload["samples"]
    if len(samples) != len(rows) or len({s["source_index"] for s in samples}) != len(rows):
        raise ValueError("Incomplete or duplicate generation records")
    by_index = {s["source_index"]: s for s in samples}
    for row in rows:
        saved = by_index[row["source_index"]]
        if any(saved.get(k) != row[k] for k in ("candidate", "target", "modification_text", "category", "split")):
            raise ValueError("Saved record is bound to another query")
        if saved.get("generation_input", {}).get("target_was_provided_to_model") is not False:
            raise ValueError("Missing query-only provenance")
    if sum(valid(s) for s in samples) / len(rows) < .9:
        raise ValueError("Structured coverage below predeclared 90% floor; no training")
    return by_index


def split_cost(ledger, split):
    with ledger.lock:
        n, charge, p, c = ledger.db.execute(
            "SELECT COUNT(*),COALESCE(SUM(charge),0),COALESCE(SUM(prompt),0),COALESCE(SUM(completion),0) FROM requests WHERE split=?", (split,)
        ).fetchone()
        states = dict(ledger.db.execute("SELECT state,COUNT(*) FROM requests WHERE split=? GROUP BY state", (split,)))
    return dict(calls=n, accounted_upper_cny=charge, prompt_tokens=p, completion_tokens=c, request_states=states)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--api-key-fd", type=int, required=True)
    args = p.parse_args()
    with os.fdopen(args.api_key_fd) as stream:
        key = stream.read(4096).strip()
    if not key:
        raise ValueError("Empty API key")
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    import fcntl
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "generation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (args.output / "generation.pid").write_text(str(os.getpid()) + "\n")
    costs = {}
    for category in CATEGORIES:
        folder = args.output / category
        folder.mkdir(exist_ok=True)
        rows_by_split = {s: load_fashioniq_queries(args.root, category, s) for s in ("train", "val")}
        if {s: len(rs) for s, rs in rows_by_split.items()} != EXPECTED[category]:
            raise ValueError("Locked original dataset counts changed")
        config = dict(version=VERSION, model=MODEL, category=category, base_url=BASE_URL,
                      fingerprints={s: fingerprint(rs) for s, rs in rows_by_split.items()},
                      prompt_sha256=hashlib.sha256((SYSTEM_PROMPT + build_user_prompt("")).encode()).hexdigest(),
                      max_tokens=1024, temperature=0, workers=8, rpm=120, known_invalid_repairs=1,
                      network_retries=0, minimum_valid_coverage=.9,
                      validation="Dress schema and deterministic program rules; not human/independent-model GT")
        ledger = Ledger(folder / "requests.sqlite3", config, max_cny=7.5,
                        max_calls=2 * sum(EXPECTED[category].values()))
        stop = threading.Event()
        rate = RateLimiter(120, stop)
        local = threading.local()
        failures = [0]
        failure_lock = threading.Lock()
        for split, rows in rows_by_split.items():
            stored = {s["source_index"]: s for s in ledger.samples(split)}
            # Validate bound records even on an explicitly resumed ledger.
            for i, sample in stored.items():
                if i >= len(rows) or any(sample.get(k) != rows[i][k] for k in ("candidate", "target", "modification_text", "category", "split")):
                    raise ValueError("Stale ledger sample")
            output = folder / f"structured_{category}_{split}.json"
            payload = dict(model=MODEL, category=category, split=split, query_sha256=fingerprint(rows),
                           query_only_generation=True, protocol=config, samples=[], state="running")

            def ask(row, phase, image, text):
                if stop.is_set():
                    raise BudgetStop("Generation stopped; no further request")
                cached = ledger.begin(split, row["source_index"], phase)
                if cached is not None:
                    return cached
                if not hasattr(local, "client"):
                    local.client = Client(key, BASE_URL, ledger)
                rate.wait()
                try:
                    answer = local.client.ask(image, text, 1024)
                    ledger.finish(answer)
                    with failure_lock:
                        failures[0] = 0
                    return answer
                except Exception as error:
                    ledger.abandon()
                    with failure_lock:
                        failures[0] += 1
                        if any(f"HTTP {c}" in str(error) for c in (401, 403, 404)) or failures[0] >= 6 or isinstance(error, BudgetStop):
                            stop.set()
                    if "HTTP 429" in str(error):
                        rate.backoff()
                    raise

            def process(row):
                require_space(args.output)
                if stop.is_set():
                    return None
                program = None
                raw = None
                errors = []
                semantic = None
                responses = []
                try:
                    image = reference_data_url(row["reference_image"])
                    prompt = SYSTEM_PROMPT + "\n" + build_user_prompt(row["modification_text"])
                    for attempt in range(2):
                        response = ask(row, "generate" if attempt == 0 else "repair_known_invalid", image, prompt)
                        responses.append(response)
                        raw = response["text"]
                        try:
                            program = extract_json_object(raw)
                            errors = validate_edit_program(program)
                        except ValueError as error:
                            errors = [str(error)]
                        if not errors:
                            program, semantic, _ = process_semantics(row, program, SimpleNamespace(semantic_repair_retries=0))
                            errors = semantic["errors"]
                        if not errors:
                            break
                        prompt = (SYSTEM_PROMPT + "\n" + build_user_prompt(row["modification_text"])
                                  + "\nCorrect only this returned invalid JSON program: " + json.dumps(program)
                                  + "\nValidation errors: " + "; ".join(errors) + ". Return full corrected JSON.")
                except Exception as error:
                    errors = [type(error).__name__ + ": " + (local.client.safe_text(str(error)) if hasattr(local, "client") else "request stopped")]
                sample = make_output_sample(row, program, raw, errors, False)
                sample.update(query_index=row["source_index"], generator_provider="qwen37-api", api_responses=responses)
                if semantic is not None:
                    sample["semantic_validation"] = semantic
                ledger.save_sample(split, sample)
                return sample

            def save(state="running"):
                payload.update(samples=[stored[i] for i in sorted(stored)], state=state,
                               completed_queries=len(stored), valid_queries=sum(valid(s) for s in stored.values()),
                               category_cost=ledger.summary(), split_cost=split_cost(ledger, split))
                atomic_write_json(output, payload)
                costs[category] = ledger.summary()
                atomic_write_json(args.output / "cost_summary.json", dict(
                    categories=costs, total_accounted_upper_cny=sum(x["accounted_upper_cny"] for x in costs.values()),
                    max_cny=15, price_checked="2026-10-22", note="Beijing list-price usage estimate; not an invoice"))

            save()
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                iterator = iter(r for r in rows if r["source_index"] not in stored)
                pending = set()
                for _ in range(8):
                    row = next(iterator, None)
                    if row is not None:
                        pending.add(pool.submit(process, row))
                last_save = time.monotonic()
                since_save = 0
                while pending:
                    done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        sample = future.result()
                        if sample is not None:
                            stored[sample["source_index"]] = sample
                            since_save += 1
                        if not stop.is_set():
                            row = next(iterator, None)
                            if row is not None:
                                pending.add(pool.submit(process, row))
                    if since_save >= 25 or time.monotonic() - last_save >= 30:
                        save()
                        print(json.dumps(dict(category=category, split=split, done=len(stored), valid=payload["valid_queries"], cost=payload["category_cost"])), flush=True)
                        last_save, since_save = time.monotonic(), 0
            save("complete" if len(stored) == len(rows) else "stopped")
            check_complete(payload, rows, category, split)
            if stop.is_set():
                raise RuntimeError("API/budget failure stopped generation; no automatic retry")
        ledger.db.close()
    print("ALL_GENERATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
