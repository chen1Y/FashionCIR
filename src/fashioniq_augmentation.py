"""Train-only FashionIQ pair proposals and reviewed metadata overlays (no new images)."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re

from fashioniq_noise import (CATEGORIES, dataset_fingerprint, file_sha, image_path,
                             object_sha, read_json)

COLORS = set("black white red blue green yellow pink purple orange brown gray grey navy beige silver gold".split())
STOP = set("a an the in on with wearing woman women man girl is has and of for to her his she he posing standing camera dress shirt top tee t".split())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def context(root, category):
    root = Path(root).resolve()
    if category not in CATEGORIES:
        raise ValueError("Unsupported category")
    # Fail closed: all category held-out splits must be available.
    heldout = set()
    for cat in CATEGORIES:
        for split in ("val", "test"):
            heldout.update(read_json(root / "image_splits" / f"split.{cat}.{split}.json"))
    train = set(read_json(root / "image_splits" / f"split.{category}.train.json"))
    rows = read_json(root / "captions" / f"cap.{category}.train.json")
    files = sorted([*root.joinpath("captions").glob("*.json"),
                    *root.joinpath("image_splits").glob("*.json")])
    signature = object_sha({str(p.relative_to(root)): file_sha(p) for p in files})
    return dict(root=root, category=category, allowed=train - heldout, rows=rows,
                existing={(r["candidate"], r["target"]) for r in rows},
                source_signature=signature, train_count=len(train), heldout_overlap=len(train & heldout))


def propose(root, category, limit=32, seed=42, noise_audit=None):
    """Cheap BLIP-text matching proposes pairs; it never certifies image semantics."""
    ctx = context(root, category)
    blocked = set()
    if noise_audit:
        audit = read_json(noise_audit)
        if audit["dataset_fingerprint"] != dataset_fingerprint(root, category):
            raise ValueError("Stale noise audit")
        # Sampling exclusion only: suspicious originals remain in the source dataset.
        blocked = {i for i, row in audit["images"].items() if row["flags"] or row["hard_errors"]}
    captions = read_json(Path(root) / "captions" / f"image_captions_{category}_train.json")
    tokens, colors, inverted = {}, {}, defaultdict(set)
    garment = {"dress": r"\b(dress|gown)\b", "shirt": r"\b(shirt|blouse)\b",
               "toptee": r"\b(top|tee|shirt|blouse|sweater)\b"}[category]
    for image_id in sorted(ctx["allowed"] - blocked):
        text = captions.get(image_id, "")
        if not isinstance(text, str) or not re.search(garment, text, re.I):
            continue
        words = set(re.findall(r"[a-z]+", text.lower()))
        color = words & COLORS
        if len(color) != 1 or image_path(root, category, image_id) is None:
            continue
        tokens[image_id] = words - STOP - COLORS
        colors[image_id] = color
        for word in tokens[image_id]:
            inverted[word].add(image_id)
    weights = {word: math.log(1 + len(tokens) / len(ids)) for word, ids in inverted.items()}
    ids = sorted(tokens)
    random.Random(seed).shuffle(ids)
    records, degree = [], Counter()
    hashes = {}
    for ref in ids:
        if len(records) >= limit:
            break
        neighbors = set().union(*(inverted[w] for w in tokens[ref])) if tokens[ref] else set()
        ranked = []
        for target in sorted(neighbors):
            if target == ref or colors[ref] == colors[target] or degree[target] >= 2:
                continue
            if (ref, target) in ctx["existing"]:
                continue
            shared = tokens[ref] & tokens[target]
            union = tokens[ref] | tokens[target]
            score = sum(weights[w] for w in shared) / max(1e-6, sum(weights[w] for w in union))
            if score >= .5:
                ranked.append((-score, target))
        for negscore, target in sorted(ranked)[:8]:
            for i in (ref, target):
                hashes.setdefault(i, file_sha(image_path(root, category, i)))
            if hashes[ref] == hashes[target]:
                continue
            records.append(dict(candidate=ref, target=target, captions=[], target_keywords=[],
                                reference_sha256=hashes[ref], target_sha256=hashes[target],
                                reference_description=captions[ref], target_description=captions[target],
                                proposal_score=round(-negscore, 6), decision="pending"))
            degree[target] += 1
            break
    return dict(schema_version=1, kind="fashioniq_triplet_additions", category=category, split="train",
                annotation_type="synthetic_pseudo_labels", source_signature=ctx["source_signature"],
                proposal_method="BLIP text overlap with a color difference; requires paired image review",
                seed=seed, original_triplets=len(ctx["rows"]), training_image_ids=ctx["train_count"],
                global_heldout_overlap_excluded=ctx["heldout_overlap"],
                audit_sampling_exclusions=len(blocked), records=records)


def validate(root, manifest, max_fraction=.1):
    ctx = context(root, manifest["category"])
    if (manifest.get("kind") != "fashioniq_triplet_additions" or manifest.get("split") != "train"
            or manifest.get("schema_version") != 1
            or manifest.get("annotation_type") != "synthetic_pseudo_labels"):
        raise ValueError("Invalid manifest schema")
    if manifest.get("source_signature") != ctx["source_signature"]:
        raise ValueError("Source metadata changed; regenerate/review proposals")
    if not 0 < max_fraction <= 1:
        raise ValueError("max_fraction must be in (0, 1]")
    accepted, seen, refs, targets = [], set(ctx["existing"]), Counter(), Counter()
    for row in manifest["records"]:
        if row.get("decision") != "accept":
            continue
        ref, target = row["candidate"], row["target"]
        pair = ref, target
        if ref == target or ref not in ctx["allowed"] or target not in ctx["allowed"] or pair in seen:
            raise ValueError(f"Not a novel train-only pair: {pair}")
        hashes = [file_sha(image_path(root, ctx["category"], i)) for i in pair]
        if None in hashes or hashes[0] == hashes[1] or hashes != [row.get("reference_sha256"), row.get("target_sha256")]:
            raise ValueError(f"Missing, duplicate or changed image: {pair}")
        caps, keywords = row.get("captions"), row.get("target_keywords")
        if (not isinstance(caps, list) or len(caps) != 2
                or any(not isinstance(t, str) or not 3 <= len(t.strip()) <= 300 for t in caps)
                or caps[0].strip().lower() == caps[1].strip().lower()
                or any(re.search(r"\b(left|right|first image|second image|photo [12])\b", t, re.I) for t in caps)):
            raise ValueError("Need two distinct standalone relative captions")
        if (not isinstance(keywords, list) or not 1 <= len(keywords) <= 8
                or any(not isinstance(t, str) or not t.strip() or len(t) > 80 for t in keywords)):
            raise ValueError("Need positive target keywords for DQU written images")
        review = row.get("review", {})
        if (not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()
                or review.get("both_garments") is not True
                or review.get("caption_satisfied") != [True, True]
                or review.get("meaningful_difference") is not True
                or review.get("target_keywords_supported") is not True
                or not isinstance(review.get("evidence"), str) or not review["evidence"].strip()):
            raise ValueError("Missing complete paired-image review; uncertainty must remain pending")
        refs[ref] += 1
        targets[target] += 1
        if refs[ref] > 2 or targets[target] > 2:
            raise ValueError("Pair degree cap exceeded")
        seen.add(pair)
        accepted.append(row)
    if not accepted or len(accepted) > max(1, int(len(ctx["rows"]) * max_fraction)):
        raise ValueError("Empty additions or augmentation budget exceeded")
    return ctx, accepted


def build_overlay(root, manifest, output, max_fraction=.1):
    ctx, accepted = validate(root, manifest, max_fraction)
    root, output = ctx["root"], Path(output).resolve()
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Overlay must be separate from original dataset")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + ".building")
    stage.mkdir()  # failed builds are retained for inspection, never reused silently
    (stage / "captions").mkdir()
    for name in ("resized_image", "image_splits"):
        (stage / name).symlink_to(root / name, target_is_directory=True)
    category = ctx["category"]
    train_name = f"cap.{category}.train.json"
    keyword_name = f"keywords_in_mods_{category}.json"
    for source in (root / "captions").iterdir():
        if source.name not in {train_name, keyword_name}:
            (stage / "captions" / source.name).symlink_to(source, target_is_directory=source.is_dir())
    additions = [{k: row[k] for k in ("candidate", "target", "captions")} for row in accepted]
    keywords = read_json(root / "captions" / keyword_name)
    for row in accepted:
        key = row["candidate"] + "_" + row["target"]
        if key in keywords:
            raise ValueError(f"Keyword collision: {key}")
        keywords[key] = [", ".join(row["target_keywords"])]
    write_json(stage / "captions" / train_name, ctx["rows"] + additions)
    write_json(stage / "captions" / keyword_name, keywords)
    write_json(stage / "augmentation_manifest.json", manifest)
    summary = dict(original_triplets=len(ctx["rows"]), added_triplets=len(additions),
                   total_triplets=len(ctx["rows"]) + len(additions), added_physical_images=0,
                   image_directory=str((stage / "resized_image").resolve()),
                   source_signature=ctx["source_signature"], category=category,
                   annotation_type="synthetic_pseudo_labels", validation_and_test="unchanged")
    write_json(stage / "augmentation_summary.json", summary)
    if context(root, category)["source_signature"] != ctx["source_signature"]:
        raise ValueError("Original metadata changed during build")
    stage.rename(output)
    return summary


def contact_sheet(root, manifest, output, start=0, count=8):
    from PIL import Image, ImageDraw, ImageOps
    rows = manifest["records"][start:start + count]
    if not rows:
        raise ValueError("No candidates for sheet")
    sheet = Image.new("RGB", (780, len(rows) * 280), "white")
    draw = ImageDraw.Draw(sheet)
    for j, row in enumerate(rows):
        for column, key in enumerate(("candidate", "target")):
            with Image.open(image_path(root, manifest["category"], row[key])) as im:
                thumb = ImageOps.contain(im.convert("RGB"), (370, 248))
            x, y = column * 390, j * 280
            sheet.paste(thumb, (x + (390 - thumb.width) // 2, y + 30))
            draw.text((x + 8, y + 8), f"{start+j}: {key} {row[key]}", fill="black")
    sheet.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("propose", "sheet", "validate", "build"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--category", choices=CATEGORIES, default="dress")
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    parser.add_argument("--noise-audit")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-fraction", type=float, default=.1)
    args = parser.parse_args()
    if args.command == "propose":
        if args.limit < 1:
            parser.error("limit must be positive")
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        write_json(args.output, propose(args.root, args.category, args.limit, args.seed, args.noise_audit))
    else:
        manifest = read_json(args.manifest)
        if args.command == "sheet":
            contact_sheet(args.root, manifest, args.output, args.start)
        elif args.command == "build":
            print(json.dumps(build_overlay(args.root, manifest, args.output, args.max_fraction)))
        else:
            _, accepted = validate(args.root, manifest, args.max_fraction)
            print(json.dumps({"accepted": len(accepted)}))


if __name__ == "__main__":
    main()
