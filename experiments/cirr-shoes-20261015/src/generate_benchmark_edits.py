"""Reference-only Qwen parsing for CIRR/Shoes, with a separate semantic self-check.

Unlike the FashionIQ generator this retains people, shoes, counts, poses and
background when requested. A Qwen self-check is a quality signal, not human GT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmark_data import BenchmarkData, read_json
from experiment_storage import require_space
from generate_structured_edits import (load_qwen, extract_json_object,
                                       validate_edit_program, atomic_write_json)


def prompt(dataset, modification):
    domain = ("All objects, people, animals, counts, poses, spatial relations and backgrounds may matter."
              if dataset == "cirr" else "Focus on footwear; preserve shoe parts, heel shape, material, color and requested context.")
    return f"""Parse the requested target state from ONLY this reference image and instruction.
{domain} Preserve negation, replacements and comparisons. Do not invent details.
Instruction: {modification}
Return JSON with exactly: retain, remove, add, relations, target_description, complexity, confidence.
retain/remove/add are lists of {{"attribute":"specific phrase", "category":"other", "region":"object or region", "evidence":"reference_image or modification_text"}}.
Use categories color/material/pattern/shape/length/fit/style/part/count/pose/viewpoint/other.
relations is a list of {{"subject":"...","relation":"...","object":"..."}}.
target_description is a concise complete intended target state.
complexity: {{"level":"simple or moderate or complex","score":0.0,"reason":"..."}}.
confidence: {{"score":0.0,"ambiguities":[]}}. Scores range from 0 to 1.
For genuinely unchanged-image instructions use retain; do not invent an edit merely to fill add/remove.
Return JSON only."""


def messages(reference, text):
    # No target ID, image, or target caption can enter this function.
    return [{"role": "user", "content": [{"type": "image", "image": reference},
                                           {"type": "text", "text": text}]}]


def response(reference, text, processor, model, device, max_tokens):
    import torch
    from PIL import Image
    with Image.open(reference) as source:
        image = source.convert("RGB")
    image.thumbnail((512, 512))
    try:
        chat = processor.apply_chat_template(messages(reference, text), tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat], images=[image], return_tensors="pt").to(device)
    finally:
        image.close()
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return processor.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("cirr", "shoes"), required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--split", choices=("train", "val"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--limit", type=int, default=20, help="New attempts, 0 means all")
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    meta = BenchmarkData(args.root, args.dataset)
    rows = meta.queries[args.split]
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    result = {"schema_version": 1, "dataset": args.dataset, "split": args.split,
              "query_sha256": fingerprint, "model": args.model, "samples": [], "state": "pending",
              "semantic_validation_method": "query-only model self-check; not human annotation"}
    if args.output.exists():
        result = read_json(args.output)
        if (result["query_sha256"] != fingerprint or result["dataset"] != args.dataset
                or result["split"] != args.split or result["model"] != args.model):
            raise ValueError("Resume output dataset/model/version mismatch")
    if args.dry_run:
        print(json.dumps({"audit": meta.audit(), "prompt_example": prompt(args.dataset, rows[0]["modification_text"]),
                          "target_sent_to_model": False}, indent=2))
        return
    if not meta.audit()["ready"]:
        raise FileNotFoundError("Dataset image audit failed; no generation started")
    require_space(args.output.parent)
    try:
        processor, model, device = load_qwen(args)
    except Exception as e:
        result.update(state="model_unavailable", error=f"{type(e).__name__}: {e}")
        atomic_write_json(args.output, result)
        raise
    stored = {sample["query_index"]: sample for sample in result["samples"]}
    attempted = 0
    for row in rows:
        idx = row["source_index"]
        if stored.get(idx, {}).get("status") == "ok":
            continue
        if args.limit and attempted >= args.limit:
            break
        require_space(args.output.parent)
        reference = str(meta.image_paths[row["candidate"]])
        sample = {**row, "query_index": idx, "category": args.dataset, "split": args.split,
                  "generation_inputs": "reference_image_and_modification_only", "status": "failed"}
        try:
            raw = response(reference, prompt(args.dataset, row["modification_text"]),
                           processor, model, device, args.max_new_tokens)
            edit = extract_json_object(raw)
            errors = [e for e in validate_edit_program(edit) if e != "at least one of add or remove must be non-empty"]
            sample.update(structured_edit=edit, validation={"valid": not errors, "errors": errors})
            if errors:
                raise ValueError("; ".join(errors))
            critic = ("Check this edit program against ONLY the reference image and instruction. "
                "Reject invented details, missing requested changes, negation/replacement errors or contradictory target properties. "
                "Do not reject people, poses, shoes or backgrounds just because they are not clothing. "
                'Return JSON {"valid":true/false,"reasons":["specific evidence"]}.\nInstruction: '
                + row["modification_text"] + "\nProgram: " + json.dumps(edit))
            judgment = extract_json_object(response(reference, critic, processor, model, device, 256))
            valid = judgment.get("valid") is True and isinstance(judgment.get("reasons"), list)
            sample["semantic_validation"] = {"valid": valid, "method": "query-only Qwen self-check", "judgment": judgment}
            if valid:
                sample["status"] = "ok"
        except Exception as e:
            sample["error"] = f"{type(e).__name__}: {e}"
        stored[idx] = sample
        attempted += 1
        result.update(samples=[stored[k] for k in sorted(stored)], state="partial")
        atomic_write_json(args.output, result)
        print(json.dumps({"query_index": idx, "status": sample["status"]}), flush=True)
    result["state"] = "complete" if len(stored) == len(rows) and all(x["status"] == "ok" for x in stored.values()) else "partial"
    atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
