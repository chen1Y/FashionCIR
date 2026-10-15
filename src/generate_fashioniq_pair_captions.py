"""Optional offline two-image Qwen labeling; refuses an occupied benchmark/GPU.

Produces synthetic labels, not human ground truth. Run only after existing experiments.
"""
import argparse
from contextlib import ExitStack
import copy
import os
from pathlib import Path
import shutil
import subprocess

from fashioniq_augmentation import context, validate, write_json
from fashioniq_noise import file_sha, image_path, read_json

GENERATE_PROMPT = """Image 1 is the REFERENCE garment; image 2 is the TARGET garment.
Create a composed-image-retrieval instruction changing garment 1 into garment 2.
Use only visible garment properties. Ignore people, pose, background and accessories.
Reject logos/charts/non-garments and ambiguous garment categories. Avoid guessing fabric,
brand or hidden features. Describe the main differences, including style and color.
Return JSON only: {"captions":["relative instruction 1","complementary relative instruction 2"],
"target_keywords":["positive visible attribute requested in the captions"]}.
The two captions together must distinguish the target from the reference; each must
be true of the target. Never mention image numbers, filenames, left or right.
If this cannot be done reliably return {"reject_reason":"reason"}."""

REVIEW_PROMPT = """Independently check a proposed retrieval instruction against both images.
Image 1 is the reference; image 2 the target. These are synthetic proposals and may be wrong.
Check garment category, direction, EVERY claimed attribute and meaningful differences.
Ignore pose/background/accessories. Each caption must describe the target correctly;
together they must express a meaningful visible change from the reference. Keywords must
be visible target attributes requested by the captions, never removed source attributes.
Use null for uncertainty. Return JSON only:
{"both_garments":true/false/null,"caption_satisfied":[true/false/null,true/false/null],
"meaningful_difference":true/false/null,"target_keywords_supported":true/false/null,
"evidence":"specific visible comparison, including any failure"}.
Proposed annotation: """


def require_idle(results, min_free_gib, output):
    """Fail closed before loading weights; never stop or compete with another job."""
    for name in ("baseline.pid", "queue.pid"):
        path = Path(results) / name
        if path.exists():
            pid = int(path.read_text().strip())
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f"Existing benchmark PID {pid} is alive; retry after the queue finishes")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
    if active:
        raise RuntimeError("GPU already has compute processes; no model loaded")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < min_free_gib * 2**30:
        raise RuntimeError("Insufficient disk reserve; no model loaded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True, help="Complete existing local Qwen3-VL checkpoint")
    parser.add_argument("--guard-results", default="/root/autodl-tmp/results/cirr_shoes_20261015")
    parser.add_argument("--min-free-gib", type=float, default=8)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--max-fraction", type=float, default=.1)
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() or Path(str(output) + ".responses.json").exists():
        raise FileExistsError("Use a new output path to preserve previous labels/responses")
    if args.limit < 1 or args.min_free_gib < 8:
        parser.error("limit must be positive; disk reserve must be at least 8 GiB")
    payload = copy.deepcopy(read_json(args.manifest))
    ctx = context(args.root, payload["category"])
    if (payload.get("kind") != "fashioniq_triplet_additions" or payload.get("schema_version") != 1
            or payload.get("split") != "train"
            or payload.get("source_signature") != ctx["source_signature"]):
        raise ValueError("Stale or non-training proposals")
    # Check boundaries BEFORE sending any image to the model, not only on export.
    for row in payload["records"]:
        pair = row["candidate"], row["target"]
        if (pair[0] == pair[1] or any(i not in ctx["allowed"] for i in pair)
                or pair in ctx["existing"]):
            raise ValueError("Proposal is not a novel train-only pair")
        hashes = [file_sha(image_path(args.root, payload["category"], i)) for i in pair]
        if (None in hashes or hashes[0] == hashes[1]
                or hashes != [row.get("reference_sha256"), row.get("target_sha256")]):
            raise ValueError("Proposal images are missing, identical or changed")
    require_idle(args.guard_results, args.min_free_gib, output)
    args.offline = True  # no downloads, including tokenizers/processors
    from PIL import Image
    import torch
    import json
    from generate_structured_edits import extract_json_object, load_qwen
    processor, model, device = load_qwen(args)

    def ask(row, prompt):
        paths = [image_path(args.root, payload["category"], row[k]) for k in ("candidate", "target")]
        if [file_sha(p) for p in paths] != [row["reference_sha256"], row["target_sha256"]]:
            raise ValueError("Changed proposal images")
        content = [{"type": "image", "image": str(p)} for p in paths] + [{"type": "text", "text": prompt}]
        text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        with ExitStack() as stack:
            images = [stack.enter_context(Image.open(p)).convert("RGB") for p in paths]
            for im in images:
                im.thumbnail((512, 512))
            inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=600, do_sample=False)
        response = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return response

    responses = []
    count = 0
    for index, row in enumerate(payload["records"]):
        if row.get("decision") != "pending" or count >= args.limit:
            continue
        count += 1
        log = dict(index=index, candidate=row["candidate"], target=row["target"])
        try:
            log["generation"] = ask(row, GENERATE_PROMPT)
            annotation = extract_json_object(log["generation"])
            if "reject_reason" in annotation:
                row.update(decision="reject", reject_reason=annotation["reject_reason"])
            else:
                log["review"] = ask(row, REVIEW_PROMPT + json.dumps(annotation))
                review = extract_json_object(log["review"])
                review["reviewer"] = f"model:{args.model}; separate prompt, same weights"
                row.update(captions=annotation["captions"], target_keywords=annotation["target_keywords"], review=review, decision="accept")
                validate(args.root, payload, args.max_fraction)
        except (ValueError, KeyError, OSError) as exc:
            row.update(decision="reject", reject_reason=f"validation: {exc}")
            log["error"] = str(exc)
        # CUDA/OOM errors propagate: do not silently continue an unreliable model session.
        responses.append(log)
        write_json(str(output) + ".responses.json", responses)
        write_json(output, payload)
        print(f"index={index} decision={row['decision']}", flush=True)


if __name__ == "__main__":
    main()
