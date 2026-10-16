"""Concurrent reference-only Shoes generation with durable metering and no secret files."""
from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter
import datetime
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import threading
import time

from benchmark_data import BenchmarkData
from experiment_storage import require_space
from generate_benchmark_edits import resume_samples
from generate_structured_edits import atomic_write_json, extract_json_object
from shoes_online_pilot import BASE_URL, MODEL, PRICE_SOURCE, Client, reference_data_url, schema_errors
from shoes_online_refine import conservative_prompt, conservative_critic

PROMPT_VERSION = "shoes-online-v3-conservative-color-gate-20261016"
COLORS = re.compile(r"\b(black|white|red|pink|blue|green|teal|grey|gray|silver|gold|golden|brown|beige|cream|purple|orange|yellow|navy|tan)\b", re.I)


def generation_prompt(modification):
    return conservative_prompt(modification) + """
For recoloring, retain the affected region's SHAPE/TEXTURE, not its obsolete colors.
Straps, laces and the toe box are part of the upper: do not retain a conflicting pink
strap or beige side panel while requiring a solid red or black/red upper.
For an opening 'above the sole' without further location, keep that relative wording;
do not decide it must be an open toe or a hole beneath the footbed.
Keep critic-independent uncertainty explicit. Complexity score measures editing difficulty,
not confidence: low for simple edits, high for complex edits. Return one complete JSON object."""


def critic_prompt(modification, edit):
    return conservative_critic(modification, edit) + """
Return at most TWO short reasons (<=20 words each), in a JSON array, even when valid.
Do not accept old upper colors conflicting with a requested final palette; straps and
toe box count as upper. An unspecified opening location must remain unspecified."""


def quality_errors(modification, edit):
    """Conservative transparent rejection, never silently edits a generated program."""
    errors = []
    retain = edit.get("retain", [])
    removed = {x["attribute"].strip().lower() for x in edit.get("remove", [])}
    for item in retain:
        if item["attribute"].strip().lower() in removed:
            errors.append("Identical attribute is both retained and removed")
    # Only final-palette assertions, not 'has pink details' or 'white design on side'.
    palette_query = re.match(r"^(?:is|are|in)\s+(?:a\s+)?(?:solid\s+)?(?:black|white|red|pink|blue|green|teal|grey|gray|silver|gold|brown|beige|cream|purple|orange|yellow|navy|tan)\b", modification, re.I)
    if palette_query:
        canonical = lambda s: {x.lower().replace("gray", "grey").replace("golden", "gold") for x in COLORS.findall(s)}
        allowed = canonical(modification)
        for item in retain:
            region = item["region"].lower()
            if re.search(r"sole|footbed|bottom|lining|interior", region) and not re.search(r"upper|strap|vamp|toe", region):
                continue
            conflict = canonical(item["attribute"]) - allowed
            if conflict:
                errors.append("Retained non-palette upper color: " + ", ".join(sorted(conflict)))
    return errors


class BudgetStop(RuntimeError):
    pass


class UncertainRequest(RuntimeError):
    pass


class Ledger:
    """Reserve before sending; settle known usage, retain unknown calls at full bound."""
    def __init__(self, path, config, max_cny=15., max_calls=23000):
        self.lock = threading.RLock()
        self.local = threading.local()
        self.max_cny, self.max_calls = max_cny, max_calls
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY, split TEXT, idx INTEGER,
          phase TEXT, state TEXT, charge REAL, prompt INTEGER, completion INTEGER, response TEXT,
          UNIQUE(split,idx,phase));
        CREATE TABLE IF NOT EXISTS samples (split TEXT, idx INTEGER, value TEXT, PRIMARY KEY(split,idx));
        """)
        encoded = json.dumps(dict(config, max_cny=max_cny, max_calls=max_calls), sort_keys=True)
        old = self.db.execute("SELECT value FROM config WHERE id=1").fetchone()
        if old and old[0] != encoded:
            self.db.close()
            raise ValueError("Online ledger protocol or budget changed; explicit new version required")
        self.db.execute("INSERT OR IGNORE INTO config VALUES (1,?)", (encoded,))
        self.db.execute("UPDATE requests SET state='unknown' WHERE state IN ('reserved','metered')")
        self.db.commit()

    def begin(self, split, idx, phase):
        with self.lock:
            row = self.db.execute("SELECT state,response FROM requests WHERE split=? AND idx=? AND phase=?", (split, idx, phase)).fetchone()
            if row:
                if row[0] == "known" and row[1]:
                    return json.loads(row[1])
                raise UncertainRequest("A prior request has no reusable response; no paid retry")
            self.local.context = split, idx, phase
            return None

    def reserve(self, max_tokens):
        with self.lock, self.db:
            calls, charge = self.db.execute("SELECT COUNT(*),COALESCE(SUM(charge),0) FROM requests").fetchone()
            reservation = (16384 * .2 + max_tokens * .8) / 1e6
            if calls >= self.max_calls or charge + reservation > self.max_cny:
                raise BudgetStop("Online generation request/cost cap reached; no further calls")
            split, idx, phase = self.local.context
            cursor = self.db.execute("INSERT INTO requests(split,idx,phase,state,charge) VALUES (?,?,?,'reserved',?)", (split, idx, phase, reservation))
            self.local.request_id = cursor.lastrowid

    def account(self, usage):
        with self.lock, self.db:
            if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("prompt_tokens", "completion_tokens")):
                return  # Missing usage keeps the full reservation.
            p, c = usage["prompt_tokens"], usage["completion_tokens"]
            self.db.execute("UPDATE requests SET state='metered',charge=?,prompt=?,completion=? WHERE id=?", ((p*.2+c*.8)/1e6, p, c, self.local.request_id))
            if p > 16384:
                raise BudgetStop("Input exceeded the per-request conservative bound")

    def finish(self, response):
        with self.lock, self.db:
            self.db.execute("UPDATE requests SET state='known',response=? WHERE id=?", (json.dumps(response), self.local.request_id))

    def abandon(self):
        request_id = getattr(self.local, "request_id", None)
        if request_id is not None:
            with self.lock, self.db:
                self.db.execute("UPDATE requests SET state='unknown' WHERE id=? AND state IN ('reserved','metered')", (request_id,))
            del self.local.request_id

    def save_sample(self, split, sample):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO samples VALUES (?,?,?)", (split, sample["query_index"], json.dumps(sample)))

    def samples(self, split):
        with self.lock:
            return [json.loads(x[0]) for x in self.db.execute("SELECT value FROM samples WHERE split=? ORDER BY idx", (split,))]

    def summary(self):
        with self.lock:
            count, charged, p, c = self.db.execute("SELECT COUNT(*),COALESCE(SUM(charge),0),COALESCE(SUM(prompt),0),COALESCE(SUM(completion),0) FROM requests").fetchone()
            unknown = self.db.execute("SELECT COUNT(*) FROM requests WHERE prompt IS NULL").fetchone()[0]
            states = dict(self.db.execute("SELECT state,COUNT(*) FROM requests GROUP BY state"))
            return dict(calls=count, max_calls=self.max_calls, max_cny=self.max_cny,
                        accounted_upper_cny=charged, estimated_list_price_cny=(p*.2+c*.8)/1e6,
                        prompt_tokens=p, completion_tokens=c, usage_missing_calls=unknown,
                        request_states=states, price_source=PRICE_SOURCE,
                        note="Known usage at Beijing list price; unknown calls retain full conservative allowance; not an invoice")


class RateLimiter:
    def __init__(self, rpm, stop):
        self.rpm, self.stop, self.next = rpm, stop, 0.
        self.lock = threading.Lock()

    def wait(self):
        while not self.stop.is_set():
            with self.lock:
                now = time.monotonic()
                if now >= self.next:
                    self.next = now + 60/self.rpm
                    return
                delay = min(1., self.next-now)
            self.stop.wait(delay)
        raise BudgetStop("Generation stopped")

    def backoff(self):
        with self.lock:
            self.rpm = max(10, self.rpm / 2)
            self.next = max(self.next, time.monotonic()+30)


def checked_complete(payload, rows, min_coverage=.5):
    stored = resume_samples(rows, payload["samples"])
    if not payload.get("attempts_complete") or len(stored) != len(rows):
        raise ValueError("Full query generation not complete")
    if sum(x["status"] == "ok" for x in stored.values()) / len(rows) < min_coverage:
        raise ValueError("Online structured coverage below predeclared quality floor")
    return stored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--api-key-fd", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rpm", type=float, default=120)
    parser.add_argument("--max-cny", type=float, default=15)
    parser.add_argument("--max-calls", type=int, default=23000)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 1 <= args.rpm <= 120 or not 0 < args.max_cny <= 15 or not 1 <= args.max_calls <= 23000:
        parser.error("Hard ceilings: 8 workers, 120 RPM, 15 CNY, 23000 requests")
    require_space(args.output, 8)
    meta = BenchmarkData(args.root, "shoes")
    rows = meta.queries[args.split]
    fingerprints = {s:hashlib.sha256(json.dumps(rs, sort_keys=True).encode()).hexdigest() for s,rs in meta.queries.items()}
    config = dict(dataset="shoes", model=MODEL, base_url=BASE_URL, prompt_version=PROMPT_VERSION,
                  fingerprints=fingerprints, generation_prompt_sha256=hashlib.sha256(generation_prompt("").encode()).hexdigest(),
                  critic_prompt_sha256=hashlib.sha256(critic_prompt("", {}).encode()).hexdigest(),
                  generation_max_tokens=768, critic_max_tokens=256, temperature=0, reference_max_side=512,
                  jpeg_quality=90, automatic_retries=0, min_valid_coverage=.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    # One generator across train/val; never reinterpret another live process's reservations.
    import fcntl
    lock = args.ledger.with_suffix(".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ledger = Ledger(args.ledger, config, args.max_cny, args.max_calls)
    stored = resume_samples(rows, ledger.samples(args.split))
    payload = dict(schema_version=1, dataset="shoes", split=args.split, model=MODEL,
                   query_sha256=fingerprints[args.split], protocol=config, samples=[], state="running",
                   generation_provider="qwen37-api", semantic_validation_method="same-model reference-only self-check; not human GT")
    stop = threading.Event()
    rate = RateLimiter(args.rpm, stop)
    failures, failure_lock = [0], threading.Lock()
    if args.api_key_fd is not None:
        with os.fdopen(args.api_key_fd, "r") as stream:
            key = stream.read(4096).strip()
    elif sys.stdin.isatty():
        key = getpass.getpass("DashScope API key (memory only): ")
    else:
        raise RuntimeError("Anonymous secret pipe or interactive secret input required")
    if not key:
        raise RuntimeError("Empty API key")
    clients, clients_lock = [], threading.Lock()
    per_thread = threading.local()

    def ask(row, phase, image, text, tokens):
        cached = ledger.begin(args.split, row["source_index"], phase)
        if cached is not None:
            return cached
        if not hasattr(per_thread, "client"):
            per_thread.client = Client(key, BASE_URL, ledger)
            with clients_lock:
                clients.append(per_thread.client)
        rate.wait()
        try:
            answer = per_thread.client.ask(image, text, tokens)
            ledger.finish(answer)
            with failure_lock:
                failures[0] = 0
            return answer
        except BudgetStop:
            stop.set()
            ledger.abandon()
            raise
        except Exception as exc:
            ledger.abandon()
            message = per_thread.client.safe_text(str(exc))
            if "HTTP 429" in message:
                rate.backoff()
            with failure_lock:
                failures[0] += 1
                fatal = failures[0] >= 8 or bool(re.search(r"HTTP (401|403|404)", message))
            if fatal:
                stop.set()
                raise BudgetStop("API authorization/model failure or repeated transport failure; inspect logs") from None
            raise RuntimeError(message) from None

    def process(row):
        require_space(args.output, 8)
        sample = dict(row, query_index=row["source_index"], category="shoes", split=args.split,
                      model=MODEL, prompt_version=PROMPT_VERSION, status="failed",
                      generation_inputs="reference_image_and_modification_only")
        try:
            image = reference_data_url(meta.image_paths[row["candidate"]])
            answer = ask(row, "generate", image, generation_prompt(row["modification_text"]), 768)
            sample["generation"] = answer
            edit = extract_json_object(answer["text"])
            errors = schema_errors(edit)
            if not errors:
                errors += quality_errors(row["modification_text"], edit)
            sample.update(structured_edit=edit, validation={"valid": not errors, "errors": errors})
            if not errors:
                critic = ask(row, "critic", image, critic_prompt(row["modification_text"], edit), 256)
                sample["critic_response"] = critic
                judgment = extract_json_object(critic["text"])
                valid = judgment.get("valid") is True and isinstance(judgment.get("reasons"), list)
                sample["semantic_validation"] = dict(valid=valid, method="same-model reference-only self-check; not human GT", judgment=judgment)
                if valid:
                    sample["status"] = "ok"
        except BudgetStop:
            raise
        except Exception as exc:
            sample["error"] = re.sub(r"sk-[A-Za-z0-9_-]{20,}", "[REDACTED]", str(exc).replace(key, "[REDACTED]"))
        ledger.save_sample(args.split, sample)
        return sample

    def save(state):
        samples = ledger.samples(args.split)
        payload.update(state=state, samples=samples, attempted_queries=len(samples), total_queries=len(rows),
                       remaining_queries=len(rows)-len(samples), attempts_complete=len(samples)==len(rows),
                       valid_queries=sum(x["status"]=="ok" for x in samples), cost=ledger.summary(),
                       updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        atomic_write_json(args.output, payload)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    futures = []
    try:
        save("running")
        for row in rows:
            if row["source_index"] not in stored:
                futures.append(executor.submit(process, row))
        completed = len(stored)
        for future in concurrent.futures.as_completed(futures):
            sample = future.result()
            completed += 1
            if completed % 10 == 0:
                save("running")
                print(json.dumps(dict(split=args.split, completed=completed, total=len(rows), cost=ledger.summary())), flush=True)
        save("complete")
        checked_complete(payload, rows)
    except BaseException as exc:
        stop.set()
        payload["error"] = re.sub(r"sk-[A-Za-z0-9_-]{20,}", "[REDACTED]", str(exc).replace(key, "[REDACTED]"))
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        save("stopped")
        raise
    finally:
        stop.set()
        executor.shutdown(wait=True, cancel_futures=True)
        for client in clients:
            client.key = ""
        key = ""
        ledger.db.close()
        lock.close()
    print(json.dumps(dict(state="complete", split=args.split, valid=payload["valid_queries"], total=len(rows), cost=payload["cost"])), flush=True)


if __name__ == "__main__":
    main()
