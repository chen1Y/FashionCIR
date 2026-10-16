"""Eight-case semantic regression check after the bounded Shoes online pilot."""
from __future__ import annotations

import argparse
import datetime
import getpass
import hashlib
import json
from pathlib import Path
import sys

from shoes_online_pilot import (BASE_URL, Budget, Client, BenchmarkData, atomic_write_json,
                                corrected_critic, diagnostics, extract_json_object,
                                optimized_prompt, reference_data_url, require_space, schema_errors)

# Selected after inspecting the first pilot: these are diagnostic, not held-out tests.
CASES = (58, 71, 236, 484, 527, 615, 1097, 1377)


def conservative_prompt(modification):
    return optimized_prompt(modification) + """
Additional conservative constraints:
- target_description describes ONLY the final intended state. Do not prefix it with the
  obsolete source state, e.g. 'a strapless shoe modified to include straps'.
- A changed property overrides conflicting retain properties in the same region.
  'solid red' cannot retain a white toe box in the upper; retaining its mesh TEXTURE is fine.
- 'Have pink in them' specifies pink presence, NOT that all existing red is removed or
  that pink must be embroidery. Keep location unspecified and record that ambiguity.
- Do not invent precise color allocation: 'black and red' need not mean black base/red accents.
- If the reference is a montage of DIFFERENT colored shoes, retain only shared visible
  structural properties. Do not pick one variant's color; record the montage ambiguity.
- A requested style such as 'closed toe stiletto' overrides contradictory old toe/heel shape.
  Do not retain the old heel type merely because it is clearly visible.
- Preserve what is specified; omit ambiguous placement, exact material and color distribution.
  Uncertainty should remain in confidence.ambiguities, not become a confident target claim.
Return JSON only for the instruction above."""


def conservative_critic(modification, edit):
    return corrected_critic(modification, edit) + """
Also check conflicts between retained and edited properties of the SAME region, global
'solid' color constraints, and unwanted preservation of a replaced style/heel/toe shape.
Do not silently fix a vague instruction by choosing exact color placement or inventing
which accent is replaced. A generic presence constraint with an explicit ambiguity is valid.
For a montage, shared structure may be retained but an arbitrary variant's color may not.
Judge edit semantics; no edited image or ground-truth target is being supplied."""


def inherited_budget(cost):
    if (cost.get("usage_missing_calls") != 0 or cost.get("max_cny") != .5
            or not 0 <= cost.get("calls", -1) <= 128
            or not 0 <= cost.get("conservative_reserved_cny", -1) <= .5):
        raise ValueError("Need a completed, metered 0.5 CNY parent pilot")
    budget = Budget(.5, 128)
    for field in ("calls", "prompt_tokens", "completion_tokens", "usage_missing_calls"):
        setattr(budget, field, cost[field])
    budget.reserved_cny = cost["conservative_reserved_cny"]
    budget.estimated_cny = cost["estimated_list_price_cny"]
    return budget


def verify_parent_rows(all_rows, selected_rows):
    # Shoes source_index belongs to the combined annotation, not its train-list offset.
    by_index = {r["source_index"]: r for r in all_rows}
    for row in selected_rows:
        current = by_index.get(row["source_index"])
        if current is None or any(row[k] != current[k]
                for k in ("candidate", "target", "modification_text", "source_index")):
            raise ValueError("Parent query mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Never overwrite or silently repeat paid regression checks")
    require_space(args.output, 8)
    raw = args.parent.read_bytes()
    parent = json.loads(raw)
    if parent["state"] != "complete" or parent["base_url"] != BASE_URL:
        raise ValueError("Parent pilot must be complete and use the verified endpoint")
    budget = inherited_budget(parent["cost"])
    meta = BenchmarkData(args.root, "shoes")
    fingerprint = hashlib.sha256(json.dumps(meta.queries["train"], sort_keys=True).encode()).hexdigest()
    if fingerprint != parent["query_sha256"]:
        raise ValueError("Dataset metadata changed")
    selected = {r["source_index"]: r for r in parent["selected_queries"]}
    rows = [selected[i] for i in CASES]
    verify_parent_rows(meta.queries["train"], rows)
    if not sys.stdin.isatty():
        raise RuntimeError("Interactive secret input required")
    client = Client(getpass.getpass("DashScope API key (memory only): "), BASE_URL, budget)
    if not client.key:
        raise RuntimeError("Empty API key")
    payload = dict(state="running", dataset="shoes", split="train", query_sha256=fingerprint,
                   parent_sha256=hashlib.sha256(raw).hexdigest(), parent_cost=parent["cost"],
                   selection="Eight post-hoc diagnostic cases, not independent validation",
                   created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), samples=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for row in rows:
            require_space(args.output, 8)
            image = reference_data_url(meta.image_paths[row["candidate"]])
            answer = client.ask(image, conservative_prompt(row["modification_text"]), 768)
            try:
                edit = extract_json_object(answer["text"])
                errors = schema_errors(edit)
            except ValueError as exc:
                edit, errors = None, [str(exc)]
            sample = dict(row, structured_edit=edit, generation=answer, status="failed",
                          query_index=row["source_index"], category="shoes", split="train",
                          validation={"valid": not errors, "errors": errors},
                          generation_inputs="reference_image_and_modification_only",
                          diagnostics=diagnostics(edit, row["modification_text"]))
            if not errors:
                critic = client.ask(image, conservative_critic(row["modification_text"], edit), 256)
                sample["critic_response"] = critic
                try:
                    judgment = extract_json_object(critic["text"])
                except ValueError:
                    judgment = {"valid": False, "reasons": ["Invalid critic JSON"]}
                valid = judgment.get("valid") is True and isinstance(judgment.get("reasons"), list)
                sample["semantic_validation"] = {"valid": valid, "judgment": judgment,
                                                 "method": "same-model self-check; not human GT"}
                if valid:
                    sample["status"] = "ok"
            payload["samples"].append(sample)
            payload["combined_cost"] = budget.summary()
            atomic_write_json(args.output, payload)
            print(json.dumps({"regression_completed": len(payload["samples"]), "combined_calls": budget.calls,
                              "combined_estimated_cny": round(budget.estimated_cny, 6)}), flush=True)
        payload["state"] = "complete"
    except Exception as exc:
        payload.update(state="stopped", error=client.safe_text(f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        client.key = ""
        payload["combined_cost"] = budget.summary()
        atomic_write_json(args.output, payload)


if __name__ == "__main__":
    main()
