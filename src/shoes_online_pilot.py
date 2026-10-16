"""Bounded, reference-only Qwen3.7-Flash prompt/critic pilot; never changes live training."""
from __future__ import annotations

import argparse
import base64
import copy
import datetime
import getpass
import hashlib
import io
import json
from pathlib import Path
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from benchmark_data import BenchmarkData
from experiment_storage import require_space
from generate_benchmark_edits import prompt as original_prompt, resume_samples
from generate_structured_edits import extract_json_object, validate_edit_program, atomic_write_json

BASE_URL = "https://llm-l8y117ub2ok0fzpk.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
ALLOWED_HOSTS = {"llm-l8y117ub2ok0fzpk.cn-beijing.maas.aliyuncs.com", "dashscope.aliyuncs.com"}
MODEL = "qwen3.7-flash"
ARMS = ("online_original", "optimized_original_critic", "optimized_corrected_critic")
PRICE_SOURCE = "https://help.aliyun.com/zh/model-studio/qwen3-7-flash"


def original_critic(modification, edit):
    # Identical to the running local benchmark critic.
    return ("Check this edit program against ONLY the reference image and instruction. "
            "Reject invented details, missing requested changes, negation/replacement errors or contradictory target properties. "
            "Do not reject people, poses, shoes or backgrounds just because they are not clothing. "
            'Return JSON {"valid":true/false,"reasons":["specific evidence"]}.\nInstruction: '
            + modification + "\nProgram: " + json.dumps(edit))


def optimized_prompt(modification):
    return """Interpret a composed footwear retrieval query. You see ONLY the reference shoe,
not the target. Describe the intended modified state, not a caption of the unchanged image.
Use only visible reference facts and requested changes. Ignore unrelated people/background.
Preserve comparisons: 'lighter' means lighter THAN THE REFERENCE, not an invented beige/white;
'fewer straps' does not imply exactly one strap. Preserve negation and replacement direction.
Retain only clear unchanged properties; never guess hidden parts, brand or exact material.
For an actual edit put the requested new/comparative state in add and/or obsolete state in remove.
For explicitly unchanged queries use retain. Keep each list <=4 concise attributes.
Return a JSON object with exactly these keys:
retain/remove/add: arrays of objects with attribute, category, region, evidence (all strings).
category MUST be ONE of: color, material, pattern, shape, length, fit, style, part, count,
pose, viewpoint, other. Do not output slash-separated alternatives.
evidence MUST be exactly reference_image OR modification_text, never the literal combined phrase.
Example add item: {"attribute":"lighter than the reference","category":"color",
"region":"shoe upper","evidence":"modification_text"}.
relations: array of {"subject":"...","relation":"...","object":"..."} or [].
target_description: short intended target description, preserving relative language when absolute
target properties are unknown. Every claim must be visible unchanged or entailed by the instruction.
complexity: {"level":"simple|moderate|complex (choose ONE)","score":0.0,"reason":"short reason"}.
confidence: {"score":0.0,"ambiguities":["uncertainties, or empty array"]}.
Use a numeric score in [0,1]. Set an actual level string such as "simple".
Return JSON only. Instruction: """ + modification


def corrected_critic(modification, edit):
    return """Audit an INTENDED EDIT using only the reference shoe and instruction.
The reference is NOT the target. A requested change should DIFFER from the reference;
never reject it merely because the reference still has the old color/heel/strap count.
Reject changes not requested, missing changes, reversed comparisons or replacement direction,
and invented precise target details. 'Lighter' does not entail beige/white; 'fewer' does not
entail exactly one. Check all retain/remove/add and target_description claims.
Visible unchanged properties can be retained. Uncertain properties should be omitted or
recorded as ambiguity, not confidently invented. No true target photo/label is available.
Return JSON {"valid":true/false,"reasons":["specific comparison to instruction and reference"]}.
Instruction: """ + modification + "\nProgram: " + json.dumps(edit)


def schema_errors(edit):
    return [e for e in validate_edit_program(edit) if e != "at least one of add or remove must be non-empty"]


def diagnostics(edit, modification):
    if not isinstance(edit, dict):
        return {"has_action_fields": False, "comparison_explicit": None, "same_add_remove": []}
    attributes = lambda k: {str(x.get("attribute", "")).strip().lower() for x in edit.get(k, []) if isinstance(x, dict)}
    comparative = re.search(r"\b(more|less|fewer|lighter|darker|higher|lower|longer|shorter|taller|bigger|smaller)\b", modification, re.I)
    return {"has_action_fields": bool(edit.get("add") or edit.get("remove")),
            "comparison_explicit": bool(re.search(r"\b(more|less|fewer|lighter|darker|higher|lower|longer|shorter|taller|bigger|smaller|than)\b", json.dumps(edit), re.I)) if comparative else None,
            "same_add_remove": sorted(attributes("add") & attributes("remove"))}


def select_queries(rows, local, limit=24, seed=42):
    if limit < 2 or limit % 2:
        raise ValueError("Use an even sample count >=2 for the balanced pilot")
    stored = resume_samples(rows, local["samples"])
    rng = random.Random(seed)
    chosen = []
    for status in ("ok", "failed"):
        pool = sorted(i for i, s in stored.items() if s["status"] == status)
        if len(pool) < limit // 2:
            raise ValueError("Not enough saved samples for both strata")
        chosen.extend(rng.sample(pool, limit // 2))
    by_index = {r["source_index"]: r for r in rows}
    return [dict(by_index[i], local_sample=stored[i]) for i in sorted(chosen)]


def request_body(reference_data_url, text, max_tokens):
    return {"model": MODEL, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": reference_data_url}},
        {"type": "text", "text": text}]}], "max_tokens": max_tokens,
        "temperature": 0, "stream": False, "enable_thinking": False,
        "response_format": {"type": "json_object"}}


class Budget:
    """Reserve each request before sending; uncertain/failed requests remain charged."""
    def __init__(self, max_cny=.5, max_calls=128):
        self.max_cny, self.max_calls = max_cny, max_calls
        self.calls = 0
        self.reserved_cny = self.estimated_cny = 0.
        self.prompt_tokens = self.completion_tokens = 0
        self.usage_missing_calls = 0

    def reserve(self, max_tokens):
        amount = 16384 * .2 / 1e6 + max_tokens * .8 / 1e6
        if self.calls >= self.max_calls or self.reserved_cny + amount > self.max_cny:
            raise RuntimeError("Pilot request/cost cap reached")
        self.calls += 1
        self.reserved_cny += amount

    def account(self, usage):
        if not isinstance(usage, dict) or not all(isinstance(usage.get(k), int) for k in ("prompt_tokens", "completion_tokens")):
            self.usage_missing_calls += 1
            return
        p, c = usage["prompt_tokens"], usage["completion_tokens"]
        self.prompt_tokens += p
        self.completion_tokens += c
        self.estimated_cny += (p * .2 + c * .8) / 1e6
        if p > 16384:
            raise RuntimeError("Usage exceeded conservative request input bound; stop")

    def summary(self):
        return dict(calls=self.calls, max_calls=self.max_calls, max_cny=self.max_cny,
                    conservative_reserved_cny=self.reserved_cny,
                    estimated_list_price_cny=self.estimated_cny,
                    prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens,
                    usage_missing_calls=self.usage_missing_calls,
                    input_cny_per_million=.2, output_cny_per_million=.8,
                    price_source=PRICE_SOURCE, price_as_of="2026-10-16",
                    note="Beijing <=32k list-price estimate; no discounts assumed, not an invoice")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Redirect refused; authorization is restricted to the configured official host")


def checked_endpoint(base_url):
    parsed = urllib.parse.urlparse(base_url)
    if (parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS or parsed.username
            or parsed.password or parsed.port not in (None, 443) or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") != "/compatible-mode/v1"):
        raise ValueError("Only the verified official Beijing endpoint is allowed")
    return base_url.rstrip("/") + "/chat/completions"


class Client:
    def __init__(self, key, base_url, budget):
        self.key, self.endpoint, self.budget = key, checked_endpoint(base_url), budget
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def safe_text(self, text):
        text = str(text).replace(self.key, "[REDACTED]") if self.key else str(text)
        return re.sub(r"sk-[A-Za-z0-9_-]{20,}", "[REDACTED]", text)

    def ask(self, image_url, text, max_tokens):
        # One <=512px image; byte upper bound plus 4096 image/format tokens stays below 16k.
        if len(text.encode("utf-8")) + 4096 > 16384:
            raise ValueError("Prompt too large for the pilot input ceiling")
        self.budget.reserve(max_tokens)
        body = json.dumps(request_body(image_url, text, max_tokens)).encode()
        request = urllib.request.Request(self.endpoint, data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + self.key})
        start = time.monotonic()
        try:
            with self.opener.open(request, timeout=60) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            # Do not print provider error bodies, request headers or credentials.
            raise RuntimeError(f"Official API HTTP {exc.code}; pilot stopped without automatic retries") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Official API transport error ({type(exc).__name__}); no automatic retry") from None
        self.budget.account(payload.get("usage"))
        choice = payload["choices"][0]
        return dict(text=self.safe_text(choice["message"].get("content") or ""),
                    usage=payload.get("usage"), model=payload.get("model"), request_id=payload.get("id"),
                    finish_reason=choice.get("finish_reason"), seconds=time.monotonic()-start)


def reference_data_url(path):
    from PIL import Image
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((512, 512))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def generation(client, row, image, improved):
    answer = client.ask(image, optimized_prompt(row["modification_text"]) if improved else original_prompt("shoes", row["modification_text"]), 768)
    try:
        edit = extract_json_object(answer["text"])
        errors = schema_errors(edit)
    except ValueError as exc:
        edit, errors = None, [str(exc)]
    return dict(generation=answer, structured_edit=edit, validation={"valid": not errors, "errors": errors})


def judge(client, row, image, generated, corrected):
    result = copy.deepcopy(generated)
    result.update({k: row[k] for k in ("candidate", "target", "modification_text", "source_index")})
    result.update(query_index=row["source_index"], category="shoes", split="train",
                  generation_inputs="reference_image_and_modification_only", generator_provider="dashscope-compatible",
                  status="failed", diagnostics=diagnostics(result["structured_edit"], row["modification_text"]))
    if result["validation"]["valid"]:
        text = (corrected_critic if corrected else original_critic)(row["modification_text"], result["structured_edit"])
        answer = client.ask(image, text, 256)
        result["critic_response"] = answer
        try:
            judgment = extract_json_object(answer["text"])
        except ValueError:
            judgment = {"valid": False, "reasons": ["Critic returned invalid JSON"]}
        valid = judgment.get("valid") is True and isinstance(judgment.get("reasons"), list)
        result["semantic_validation"] = {"valid": valid, "method": "same-model reference-only self-check; not human GT", "judgment": judgment}
        if valid:
            result["status"] = "ok"
    return result


def summarize(payload):
    output = {}
    for arm, samples in payload["arms"].items():
        output[arm] = dict(samples=len(samples), schema_valid=sum(x.get("validation", {}).get("valid", False) for x in samples),
                           selfcheck_ok=sum(x.get("status")=="ok" for x in samples),
                           has_action_fields=sum(diagnostics(x.get("structured_edit"), x["modification_text"])["has_action_fields"] for x in samples))
    payload["summary"] = output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--local-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cny", type=float, default=.5)
    parser.add_argument("--max-calls", type=int, default=128)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    checked_endpoint(args.base_url)
    if not 2 <= args.limit <= 48 or args.limit % 2 or not 0 < args.max_cny <= 1 or not 1 <= args.max_calls <= 256:
        parser.error("Pilot limits: even sample count 2..48, cost cap (0,1] CNY, calls 1..256")
    if args.output.exists():
        raise FileExistsError("Use a new output directory; paid calls are never silently repeated")
    require_space(args.output, 8)
    meta = BenchmarkData(args.root, "shoes")
    raw = args.local_results.read_bytes()
    local = json.loads(raw)
    rows = meta.queries["train"]
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    if local.get("dataset") != "shoes" or local.get("split") != "train" or local.get("query_sha256") != fingerprint:
        raise ValueError("Local comparison snapshot does not match Shoes train metadata")
    selected = select_queries(rows, local, args.limit, args.seed)
    args.output.mkdir(parents=True)
    payload = dict(schema_version=1, state="prepared", model=MODEL, base_url=args.base_url,
                   created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   dataset="shoes", split="train", query_sha256=fingerprint,
                   local_snapshot_sha256=hashlib.sha256(raw).hexdigest(), local_snapshot_records=len(local["samples"]),
                   seed=args.seed, selection="Equal saved local ok/failed strata; challenge pilot, not population accuracy",
                   selected_queries=[{k:r[k] for k in ("candidate","target","modification_text","source_index")} for r in selected],
                   arms={"local_8b": [r["local_sample"] for r in selected], **{arm:[] for arm in ARMS}},
                   third_arm_reuses_second_arm_generation=True, raw_images_persisted=False,
                   requested_max_calls=args.max_calls, requested_max_cny=args.max_cny,
                   inference=dict(temperature=0, enable_thinking=False, response_format="json_object",
                                  generation_max_tokens=768, critic_max_tokens=256,
                                  reference_max_side=512, jpeg_quality=90, automatic_retries=0),
                   annotation_note="Model self-checks are not independent human accuracy; no retrieval training or protocol change")
    result_path = args.output / "pilot.json"
    summarize(payload)
    atomic_write_json(result_path, payload)
    if args.prepare_only:
        print(json.dumps(payload["summary"]))
        return
    if not sys.stdin.isatty():
        raise RuntimeError("Interactive terminal required for secret input; no key files or command arguments")
    key = getpass.getpass("DashScope API key (memory only): ")
    if not key:
        raise RuntimeError("Empty API key")
    budget = Budget(args.max_cny, args.max_calls)
    client = Client(key, args.base_url, budget)
    key = ""
    payload["state"] = "running"
    try:
        for row in selected:
            image = reference_data_url(meta.image_paths[row["candidate"]])
            a = generation(client, row, image, False)
            payload["arms"][ARMS[0]].append(judge(client, row, image, a, False))
            b = generation(client, row, image, True)
            payload["arms"][ARMS[1]].append(judge(client, row, image, b, False))
            payload["arms"][ARMS[2]].append(judge(client, row, image, b, True))
            summarize(payload)
            payload["cost"] = budget.summary()
            atomic_write_json(result_path, payload)
            print(json.dumps({"completed_queries":len(payload["arms"][ARMS[2]]),"calls":budget.calls,
                              "estimated_cny":round(budget.estimated_cny,6)}),flush=True)
        payload["state"] = "complete"
    except Exception as exc:
        payload.update(state="stopped", error=client.safe_text(f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        payload["cost"] = budget.summary()
        summarize(payload)
        atomic_write_json(result_path, payload)
        client.key = ""
    print(json.dumps({"state":payload["state"],"summary":payload["summary"],"cost":payload["cost"]}))


if __name__ == "__main__":
    main()
