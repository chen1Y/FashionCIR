"""Generate query-only structured edit programs for FashionIQ with Qwen3-VL.

The target image and target identifier are deliberately never passed to Qwen.
They are retained only as output metadata for later retrieval evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"
ATTRIBUTE_CATEGORIES = {
    "color",
    "material",
    "pattern",
    "shape",
    "length",
    "fit",
    "style",
    "part",
    "count",
    "pose",
    "viewpoint",
    "other",
}
COMPLEXITY_LEVELS = {"simple", "moderate", "complex"}

SYSTEM_PROMPT = """You are a visual edit parser for composed image retrieval.
Given ONLY a reference image and a natural-language modification instruction,
convert the requested change into a structured JSON edit program.

Rules:
1. Describe the intended target state, not the current source state.
2. Separate properties that must be retained, removed/replaced, and added.
3. Preserve negation, comparison, spatial relation, and replacement semantics.
4. Do not invent attributes that cannot be inferred from the image or instruction.
5. Use concise English phrases.
6. Return one JSON object only. Do not use Markdown or explanatory prose.
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate structured FashionIQ edit JSON with Qwen3-VL."
    )
    parser.add_argument(
        "--fashioniq-root",
        type=Path,
        default=Path("../data/FashionIQ"),
        help="FashionIQ root containing captions/ and resized_image/.",
    )
    parser.add_argument(
        "--category",
        choices=("dress", "shirt", "toptee"),
        default="dress",
    )
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Hugging Face model identifier or local model directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON. Defaults to captions/structured_edits_<category>_<split>.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum new samples to process. Use 0 for all samples.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Load model and processor from local cache only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect selected records and prompts without loading Qwen or writing output.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing output file without loading Qwen.",
    )
    parser.add_argument(
        "--omit-raw-response",
        action="store_true",
        help="Do not store the raw Qwen response for successful samples.",
    )
    return parser.parse_args()


def normalize_instruction(captions: Any) -> str:
    if isinstance(captions, str):
        return " ".join(captions.split())
    if isinstance(captions, list):
        parts = [" ".join(str(part).split()) for part in captions if str(part).strip()]
        return " and ".join(parts)
    raise TypeError(f"Unsupported captions value: {type(captions).__name__}")


def sample_id(category: str, split: str, index: int, item: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "category": category,
            "split": split,
            "index": index,
            "candidate": item.get("candidate"),
            "captions": item.get("captions"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"fashioniq:{category}:{split}:{index}:{digest}"


def load_fashioniq_queries(
    root: Path, category: str, split: str
) -> list[dict[str, Any]]:
    caption_path = root / "captions" / f"cap.{category}.{split}.json"
    if not caption_path.is_file():
        raise FileNotFoundError(f"Caption file not found: {caption_path}")

    with caption_path.open("r", encoding="utf-8") as handle:
        raw_items = json.load(handle)
    if not isinstance(raw_items, list):
        raise ValueError(f"Expected a JSON list in {caption_path}")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} is not a JSON object")
        candidate = str(item["candidate"])
        image_path = root / "resized_image" / f"{candidate}.jpg"
        records.append(
            {
                "query_id": sample_id(category, split, index, item),
                "dataset": "fashioniq",
                "category": category,
                "split": split,
                "source_index": index,
                "candidate": candidate,
                # Kept for evaluation metadata only. Never used in build_messages().
                "target": str(item.get("target", "")),
                "reference_image": str(image_path.resolve()),
                "modification_text": normalize_instruction(item["captions"]),
            }
        )
    return records


def build_user_prompt(modification_text: str) -> str:
    return f"""Inspect the reference image and parse this modification instruction:

{modification_text}

Return a JSON object with these seven required top-level fields:
- "retain": list of visible source properties that remain valid in the target.
- "remove": list of source properties explicitly removed or replaced.
- "add": list of desired target properties introduced by the instruction.
- "relations": list of semantic relations such as replaces, less_than, more_than,
  above, below, without, or instead_of.
- "target_description": a non-empty, concise description of the resulting target.
- "complexity": object with level, score, and reason.
- "confidence": object with score and ambiguities.

Every item in retain/remove/add must contain:
{{"attribute": "specific phrase", "category": "allowed category",
  "region": "affected garment region", "evidence": "reference_image or modification_text"}}

Every relation must contain:
{{"subject": "specific phrase", "relation": "specific relation",
  "object": "specific phrase"}}

The "complexity" object must contain:
- "level": one of simple, moderate, or complex;
- "score": a calibrated number between 0 and 1, where larger means more complex;
- "reason": a specific explanation.

The "confidence" object must contain:
- "score": a calibrated number between 0 and 1, where larger means more certain;
- "ambiguities": a list of specific unresolved issues, or an empty list.

Allowed category values:
{", ".join(sorted(ATTRIBUTE_CATEGORIES))}

Mandatory requirements:
1. Inspect the image and identify the main garment or object.
2. At least one of "add" or "remove" must be non-empty.
3. "target_description" must be non-empty and reflect every requested change.
4. For comparative words such as longer, lighter, or less revealing, preserve the
   comparison even if an exact absolute value cannot be inferred.
5. Do not return placeholder values such as "string", "attribute", or empty text.
6. If uncertain, record the issue in confidence.ambiguities instead of returning
   an empty edit program.
7. Return JSON only.
"""


def build_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    # Deliberately access only query-side fields here. In particular, target is forbidden.
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": record["reference_image"]},
                {
                    "type": "text",
                    "text": build_user_prompt(record["modification_text"]),
                },
            ],
        },
    ]


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()

    for position, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("No valid JSON object found in model response")


def validate_attribute_item(
    value: Any, path: str, errors: list[str]
) -> None:
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object")
        return
    for key in ("attribute", "category", "region", "evidence"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            errors.append(f"{path}.{key} must be a non-empty string")
    category = value.get("category")
    if isinstance(category, str) and category not in ATTRIBUTE_CATEGORIES:
        errors.append(f"{path}.category has unsupported value: {category}")
    evidence = value.get("evidence")
    if evidence not in {"reference_image", "modification_text"}:
        errors.append(f"{path}.evidence must be reference_image or modification_text")


def validate_edit_program(program: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(program, dict):
        return ["structured_edit must be an object"]

    expected_keys = {
        "retain",
        "remove",
        "add",
        "relations",
        "target_description",
        "complexity",
        "confidence",
    }
    missing = expected_keys - set(program)
    if missing:
        errors.append(f"missing keys: {', '.join(sorted(missing))}")

    for field in ("retain", "remove", "add"):
        values = program.get(field)
        if not isinstance(values, list):
            errors.append(f"{field} must be a list")
            continue
        for index, value in enumerate(values):
            validate_attribute_item(value, f"{field}[{index}]", errors)

    relations = program.get("relations")
    if not isinstance(relations, list):
        errors.append("relations must be a list")
    else:
        for index, relation in enumerate(relations):
            if not isinstance(relation, dict):
                errors.append(f"relations[{index}] must be an object")
                continue
            for key in ("subject", "relation", "object"):
                if not isinstance(relation.get(key), str) or not relation[key].strip():
                    errors.append(f"relations[{index}].{key} must be a non-empty string")

    if not isinstance(program.get("target_description"), str) or not program[
        "target_description"
    ].strip():
        errors.append("target_description must be a non-empty string")
    add_values = program.get("add")
    remove_values = program.get("remove")
    if (
        isinstance(add_values, list)
        and isinstance(remove_values, list)
        and not add_values
        and not remove_values
    ):
        errors.append("at least one of add or remove must be non-empty")

    complexity = program.get("complexity")
    if not isinstance(complexity, dict):
        errors.append("complexity must be an object")
    else:
        if complexity.get("level") not in COMPLEXITY_LEVELS:
            errors.append("complexity.level must be simple, moderate, or complex")
        score = complexity.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            errors.append("complexity.score must be numeric")
        elif not 0 <= float(score) <= 1:
            errors.append("complexity.score must be between 0 and 1")
        if not isinstance(complexity.get("reason"), str):
            errors.append("complexity.reason must be a string")

    confidence = program.get("confidence")
    if not isinstance(confidence, dict):
        errors.append("confidence must be an object")
    else:
        score = confidence.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            errors.append("confidence.score must be numeric")
        elif not 0 <= float(score) <= 1:
            errors.append("confidence.score must be between 0 and 1")
        ambiguities = confidence.get("ambiguities")
        if not isinstance(ambiguities, list) or not all(
            isinstance(item, str) for item in ambiguities
        ):
            errors.append("confidence.ambiguities must be a list of strings")
    return errors


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)


def load_existing_output(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "generator": {},
            "samples": [],
        }
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(f"Invalid output container: {path}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema version {payload.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    return payload


def validate_output_file(path: Path) -> int:
    payload = load_existing_output(path)
    total = len(payload["samples"])
    invalid = 0
    failed = 0
    for sample in payload["samples"]:
        if sample.get("status") != "ok":
            failed += 1
            continue
        errors = validate_edit_program(sample.get("structured_edit"))
        if errors:
            invalid += 1
            print(f"{sample.get('query_id')}: {'; '.join(errors)}")
    print(f"validated={total} valid={total-invalid-failed} invalid={invalid} failed={failed}")
    return 1 if invalid else 0


def choose_dtype(torch_module: Any) -> Any:
    if not torch_module.cuda.is_available():
        return torch_module.float32
    if torch_module.cuda.is_bf16_supported():
        return torch_module.bfloat16
    return torch_module.float16


def load_qwen(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    dtype = choose_dtype(torch)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        local_files_only=args.offline,
    )
    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=args.offline,
    )
    model.eval()

    input_device = next(
        parameter.device
        for parameter in model.parameters()
        if parameter.device.type != "meta"
    )
    return processor, model, input_device


def generate_response(
    record: dict[str, Any],
    processor: Any,
    model: Any,
    input_device: Any,
    max_new_tokens: int,
    correction: str | None = None,
) -> str:
    import torch
    from PIL import Image

    image_path = Path(record["reference_image"])
    if not image_path.is_file():
        raise FileNotFoundError(f"Reference image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    messages = build_messages(record)
    if correction:
        messages[-1]["content"][-1]["text"] += (
            "\n\nYour previous response failed validation:\n"
            f"{correction}\n"
            "Regenerate the entire JSON object from scratch and fix every listed error."
        )
    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text_input],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(input_device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def make_output_sample(
    record: dict[str, Any],
    structured_edit: dict[str, Any] | None,
    raw_response: str | None,
    errors: list[str],
    omit_raw_response: bool,
) -> dict[str, Any]:
    sample = {
        **record,
        "generation_input": {
            "reference_image": record["reference_image"],
            "modification_text": record["modification_text"],
            "target_was_provided_to_model": False,
        },
        "status": "ok" if structured_edit is not None and not errors else "failed",
        "structured_edit": structured_edit,
        "validation": {
            "valid": structured_edit is not None and not errors,
            "errors": errors,
        },
    }
    if raw_response is not None and (errors or not omit_raw_response):
        sample["raw_response"] = raw_response
    return sample


def run_generation(args: argparse.Namespace) -> int:
    root = args.fashioniq_root.resolve()
    output = (
        args.output.resolve()
        if args.output
        else root
        / "captions"
        / f"structured_edits_{args.category}_{args.split}.json"
    )
    records = load_fashioniq_queries(root, args.category, args.split)
    records = records[args.start_index :]

    if args.dry_run:
        selected = records if args.limit == 0 else records[: args.limit]
        for record in selected[:3]:
            preview = {
                key: value
                for key, value in record.items()
                if key != "target"
            }
            preview["messages"] = build_messages(record)
            print(json.dumps(preview, ensure_ascii=False, indent=2))
        print(
            f"dry-run selected={len(selected)}; "
            "target metadata was excluded from every model message"
        )
        return 0

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
        return 0

    processor, model, input_device = load_qwen(args)
    processed = 0
    for record in pending:
        raw_response: str | None = None
        structured_edit: dict[str, Any] | None = None
        errors: list[str] = []
        correction: str | None = None

        for attempt in range(args.max_retries + 1):
            try:
                raw_response = generate_response(
                    record,
                    processor,
                    model,
                    input_device,
                    args.max_new_tokens,
                    correction,
                )
                structured_edit = extract_json_object(raw_response)
                errors = validate_edit_program(structured_edit)
                if not errors:
                    break
            except Exception as exc:
                errors = [f"{type(exc).__name__}: {exc}"]
            correction = "; ".join(errors)
            if attempt < args.max_retries:
                print(
                    f"retry query={record['query_id']} "
                    f"attempt={attempt + 2} errors={errors}",
                    file=sys.stderr,
                )

        sample = make_output_sample(
            record,
            structured_edit,
            raw_response,
            errors,
            args.omit_raw_response,
        )
        existing_by_id[record["query_id"]] = sample
        processed += 1
        print(
            f"[{processed}/{len(pending)}] {record['query_id']} "
            f"status={sample['status']}"
        )

        if processed % args.checkpoint_every == 0 or processed == len(pending):
            existing["generator"] = {
                "model": args.model,
                "dataset": "fashioniq",
                "category": args.category,
                "split": args.split,
                "query_only_generation": True,
            }
            existing["samples"] = sorted(
                existing_by_id.values(),
                key=lambda item: item.get("source_index", -1),
            )
            atomic_write_json(output, existing)
            print(f"checkpoint: {output}")

    return validate_output_file(output)


def main() -> int:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be zero or positive")
    if args.start_index < 0:
        raise ValueError("--start-index must be zero or positive")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be zero or positive")

    output = (
        args.output.resolve()
        if args.output
        else args.fashioniq_root.resolve()
        / "captions"
        / f"structured_edits_{args.category}_{args.split}.json"
    )
    if args.validate_only:
        return validate_output_file(output)
    return run_generation(args)


if __name__ == "__main__":
    raise SystemExit(main())
