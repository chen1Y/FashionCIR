#!/usr/bin/env python3
"""Generate FashionIQ structured edits with an OpenAI-compatible DashScope API.

The output name is provider-specific by default, so this script never overwrites
the local Qwen3-VL files produced by generate_structured_edits.py.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from generate_structured_edits import (
    SYSTEM_PROMPT,
    atomic_write_json,
    build_user_prompt,
    extract_json_object,
    load_existing_output,
    load_fashioniq_queries,
    make_output_sample,
    process_semantics,
    validate_edit_program,
    validate_output_file,
)


DEFAULT_BASE_URL = (
    "https://llm-l8y117ub2ok0fzpk.cn-beijing.maas.aliyuncs.com/"
    "compatible-mode/v1"
)
_thread_local = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate structured FashionIQ edits using Qwen3.7-Flash."
    )
    parser.add_argument(
        "--fashioniq-root", type=Path, default=Path("../data/FashionIQ")
    )
    parser.add_argument(
        "--category", choices=("dress", "shirt", "toptee"), default="dress"
    )
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--model", default="qwen3.7-flash")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Provider-specific output path. Defaults to "
            "structured_edits_<category>_<split>_qwen37flash.json."
        ),
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable model reasoning. Disabled by default for JSON generation efficiency.",
    )
    parser.add_argument("--omit-raw-response", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def get_client(args: argparse.Namespace) -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        client = OpenAI(
            api_key=api_key,
            base_url=args.base_url,
            timeout=args.timeout,
            max_retries=0,
        )
        _thread_local.client = client
    return client


def image_data_url(path_value: str) -> str:
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"Reference image not found: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def request_once(
    args: argparse.Namespace,
    record: dict[str, Any],
    correction: str | None = None,
    previous_program: dict[str, Any] | None = None,
) -> str:
    prompt = build_user_prompt(record["modification_text"])
    if correction:
        previous_json = (
            json.dumps(previous_program, ensure_ascii=False)
            if previous_program is not None
            else "null"
        )
        prompt += (
            "\nYour previous output failed validation: "
            f"{correction}\nPrevious JSON: {previous_json}\n"
            "Return the entire corrected JSON object, and nothing else."
        )
    completion = get_client(args).chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url(record["reference_image"])
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        max_tokens=args.max_tokens,
        stream=False,
        extra_body={"enable_thinking": args.enable_thinking},
    )
    if not completion.choices:
        raise RuntimeError("API response contained no choices")
    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("API response content was empty")
    return content.strip()


def process_record(
    args: argparse.Namespace, record: dict[str, Any]
) -> dict[str, Any]:
    raw_response: str | None = None
    structured_edit: dict[str, Any] | None = None
    errors: list[str] = []

    for attempt in range(args.max_retries + 1):
        try:
            raw_response = request_once(
                args,
                record,
                correction="; ".join(errors) if errors else None,
                previous_program=structured_edit,
            )
            structured_edit = extract_json_object(raw_response)
            errors = validate_edit_program(structured_edit)
            if not errors:
                structured_edit, semantic_validation, _ = process_semantics(
                    record, structured_edit, args
                )
                errors = list(semantic_validation["errors"])
                if not errors:
                    sample = make_output_sample(
                        record,
                        structured_edit,
                        raw_response,
                        errors,
                        args.omit_raw_response,
                    )
                    sample["semantic_validation"] = semantic_validation
                    sample["generator_provider"] = "dashscope-compatible"
                    return sample
        except Exception as exc:
            errors = [f"{type(exc).__name__}: {exc}"]

        if attempt < args.max_retries:
            delay = min(30.0, (2**attempt) + random.random())
            time.sleep(delay)

    sample = make_output_sample(
        record,
        structured_edit,
        raw_response,
        errors,
        args.omit_raw_response,
    )
    sample["generator_provider"] = "dashscope-compatible"
    return sample


def main() -> int:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be zero or positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")

    root = args.fashioniq_root.resolve()
    output = (
        args.output.resolve()
        if args.output
        else root
        / "captions"
        / f"structured_edits_{args.category}_{args.split}_qwen37flash.json"
    )
    if args.validate_only:
        return validate_output_file(output)
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    records = load_fashioniq_queries(root, args.category, args.split)
    records = records[args.start_index :]
    existing = load_existing_output(output)
    existing_by_id = {
        sample["query_id"]: sample
        for sample in existing["samples"]
        if sample.get("query_id")
    }
    pending = [
        record
        for record in records
        if record["query_id"] not in existing_by_id
        or existing_by_id[record["query_id"]].get("status") != "ok"
    ]
    if args.limit > 0:
        pending = pending[: args.limit]
    if not pending:
        print(f"No pending samples. Output is up to date: {output}")
        return validate_output_file(output)

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_record = {
            executor.submit(process_record, args, record): record
            for record in pending
        }
        for future in as_completed(future_to_record):
            record = future_to_record[future]
            try:
                sample = future.result()
            except Exception as exc:
                sample = make_output_sample(
                    record,
                    None,
                    None,
                    [f"{type(exc).__name__}: {exc}"],
                    args.omit_raw_response,
                )
                sample["generator_provider"] = "dashscope-compatible"
            existing_by_id[record["query_id"]] = sample
            completed += 1
            print(
                f"[{completed}/{len(pending)}] index={record['source_index']} "
                f"status={sample['status']}",
                flush=True,
            )
            if (
                completed % args.checkpoint_every == 0
                or completed == len(pending)
            ):
                existing["generator"] = {
                    "model": args.model,
                    "provider": "dashscope-compatible",
                    "base_url": args.base_url,
                    "dataset": "fashioniq",
                    "category": args.category,
                    "split": args.split,
                    "query_only_generation": True,
                    "enable_thinking": args.enable_thinking,
                }
                existing["samples"] = sorted(
                    existing_by_id.values(),
                    key=lambda item: item.get("source_index", -1),
                )
                atomic_write_json(output, existing)
                print(f"checkpoint: {output}", flush=True)

    return validate_output_file(output)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
