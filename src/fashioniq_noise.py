"""Conservative FashionIQ audits and dataset-bound, train-only exclusion policies."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


CATEGORIES = ("dress", "shirt", "toptee")
SAFE_ID = re.compile(r"[A-Za-z0-9_-]+\Z")
GARMENT_WORD = re.compile(r"\b(dress|shirt|tee|top|garment|clothing|blouse|sweater)\b", re.I)
DOMAIN_CLAIM = re.compile(
    r"\b(?:not (?:a |an )?(?:dress|shirt|garment|piece of clothing)|"
    r"company (?:name|logo)|product is text|(?:only|just) (?:a )?(?:logo|text)|"
    r"(?:is|shows) (?:a |an )?(?:logo|brand name))\b", re.I)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    if path is None or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_path(root, category, image_id):
    if not isinstance(image_id, str) or not SAFE_ID.fullmatch(image_id):
        return None
    base = Path(root) / "resized_image"
    for path in (base / f"{image_id}.jpg", base / category / f"{image_id}.jpg"):
        if path.is_file():
            return path
    return None


def dataset_fingerprint(root, category, split="train"):
    if category not in CATEGORIES or split not in {"train", "val"}:
        raise ValueError("Unsupported category/split")
    root = Path(root)
    paths = [root / "captions" / f"cap.{category}.{split}.json",
             root / "image_splits" / f"split.{category}.{split}.json"]
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("Missing FashionIQ caption or image split JSON")
    paths.append(root / "captions" / f"correction_dict_{category}.json")
    return object_sha({p.name: file_sha(p) for p in paths})


def annotation_errors(row, split_ids):
    if not isinstance(row, dict):
        return ["invalid_annotation_schema"]
    errors = []
    for key in ("candidate", "target"):
        value = row.get(key)
        if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
            errors.append(f"invalid_{key}_id")
        elif value not in split_ids:
            errors.append(f"{key}_outside_split")
    captions = row.get("captions")
    if not isinstance(captions, list) or len(captions) != 2 or any(not isinstance(t, str) for t in captions):
        errors.append("invalid_caption_schema")
    return errors


def text_flags(captions):
    flags = []
    if not all(t.strip() for t in captions):
        flags.append("empty_caption")
    claims = [bool(DOMAIN_CLAIM.search(t)) for t in captions]
    if any(claims):
        flags.append("caption_non_garment_claim")
        if any(not claim and GARMENT_WORD.search(t) for claim, t in zip(claims, captions)):
            flags.append("caption_domain_disagreement")
    # Deliberately narrow: two explicit solid/pure colors. This is still only
    # a review hint; two captions may legitimately mention different parts.
    colors = [set(re.findall(r"\b(?:solid|pure|entirely) (black|white|red|blue|green|yellow|pink|purple)\b",
                             t.lower())) for t in captions]
    if all(colors) and colors[0].isdisjoint(colors[1]):
        flags.append("caption_color_disagreement")
    return flags


def inspect_image(path):
    from PIL import Image, ImageStat
    result = {"path": str(path) if path else None, "sha256": None, "flags": [], "hard_errors": []}
    if path is None:
        result["hard_errors"].append("missing_image")
        return result
    try:
        result["sha256"] = file_sha(path)
        with Image.open(path) as image:
            image.load()  # verify() alone does not decode all pixel data
            result["width"], result["height"] = image.size
            gray = image.convert("L").resize((64, 64))
            result["gray_std"] = ImageStat.Stat(gray).stddev[0]
            result["entropy"] = gray.entropy()
            if min(image.size) < 32:
                result["flags"].append("very_small_image")
            if result["gray_std"] < 2 or result["entropy"] < .1:
                result["flags"].append("low_information_image")
            pixels = list(image.convert("L").resize((9, 8)).getdata())
            bits = [pixels[y * 9 + x] > pixels[y * 9 + x + 1] for y in range(8) for x in range(8)]
            result["dhash"] = f"{sum(int(bit) << i for i, bit in enumerate(bits)):016x}"
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        result["hard_errors"].append("unreadable_image")
        result["error_type"] = type(exc).__name__
    return result


def audit(root, category, split="train", limit_images=0):
    root = Path(root).resolve()
    raw = read_json(root / "captions" / f"cap.{category}.{split}.json")
    split_ids = set(read_json(root / "image_splits" / f"split.{category}.{split}.json"))
    ids = set(split_ids)
    for row in raw:
        if isinstance(row, dict):
            ids.update(v for key in ("candidate", "target")
                       if isinstance((v := row.get(key)), str) and SAFE_ID.fullmatch(v))
    selected = sorted(ids)[:limit_images] if limit_images else sorted(ids)
    images = {i: dict(image_id=i, **inspect_image(image_path(root, category, i))) for i in selected}
    captions_path = root / "captions" / f"image_captions_{category}_{split}.json"
    generated = read_json(captions_path) if captions_path.exists() else {}
    groups = defaultdict(list)
    for i, entry in images.items():
        caption = generated.get(i, "")
        if isinstance(caption, str) and re.search(r"\b(logo|text|brand name)\b", caption, re.I) and not GARMENT_WORD.search(caption):
            entry["flags"].append("generated_caption_non_garment_hint")
            entry["generated_caption"] = caption
        if entry["sha256"]:
            groups[entry["sha256"]].append(i)
    duplicates = [group for group in groups.values() if len(group) > 1]
    for group in duplicates:
        for i in group:
            images[i]["flags"].append("byte_identical_images")
    report = dict(schema_version=1, category=category, split=split, root=str(root),
                  dataset_fingerprint=dataset_fingerprint(root, category, split),
                  partial=len(selected) != len(ids), total_images=len(ids), images=images,
                  exact_duplicate_groups=duplicates, total_queries=len(raw))
    refresh_queries(report, raw, split_ids)
    return report


def refresh_queries(report, raw, split_ids):
    queries = []
    seen = {}
    for i, row in enumerate(raw):
        hard = annotation_errors(row, split_ids)
        flags = [] if hard else text_flags(row["captions"])
        if isinstance(row, dict):
            if row.get("candidate") == row.get("target"):
                flags.append("reference_equals_target")
            for role in ("candidate", "target"):
                image_id = row.get(role)
                image = report["images"].get(image_id, {}) if isinstance(image_id, str) else {}
                hard.extend(f"{role}:{e}" for e in image.get("hard_errors", []))
                flags.extend(f"{role}:{f}" for f in image.get("flags", []))
        signature = object_sha(row)
        if signature in seen:
            flags.append(f"duplicate_triplet_of_{seen[signature]}")
        seen.setdefault(signature, i)
        if flags or hard:
            queries.append(dict(source_index=i, row_sha256=signature, annotation=row,
                                flags=sorted(set(flags)), hard_errors=sorted(set(hard))))
    report["queries"] = queries
    report["summary"] = dict(images_scanned=len(report["images"]),
        images_flagged=sum(bool(v["flags"] or v["hard_errors"]) for v in report["images"].values()),
        unusable_images=sum(bool(v["hard_errors"]) for v in report["images"].values()),
        queries_flagged=len(queries), queries_with_hard_errors=sum(bool(q["hard_errors"]) for q in queries),
        query_reason_counts=dict(Counter(f for q in queries for f in q["flags"] + q["hard_errors"])))


def review_template(report):
    return dict(schema_version=1, category=report["category"], split=report["split"],
                dataset_fingerprint=report["dataset_fingerprint"],
                images=[dict(image_id=i, sha256=r["sha256"], decision="pending", reviewer="", reason="")
                        for i, r in report["images"].items() if r["flags"] or r["hard_errors"]],
                queries=[dict(source_index=q["source_index"], row_sha256=q["row_sha256"],
                              decision="pending", reviewer="", reason="") for q in report["queries"]])


def compile_policy(root, report, reviews=None, max_exclusion_fraction=.2):
    category = report["category"]
    if report["split"] != "train" or report["partial"]:
        raise ValueError("Only a complete train audit can produce a training filter")
    fingerprint = dataset_fingerprint(root, category)
    if report["dataset_fingerprint"] != fingerprint:
        raise ValueError("Stale audit: dataset fingerprint changed")
    raw = read_json(Path(root) / "captions" / f"cap.{category}.train.json")
    train_ids = set(read_json(Path(root) / "image_splits" / f"split.{category}.train.json"))
    bad_images = {i: dict(image_id=i, sha256=r["sha256"], reasons=r["hard_errors"],
                         source="unusable_file") for i, r in report["images"].items()
                  if r["hard_errors"] and i in train_ids}
    bad_queries = {q["source_index"]: dict(source_index=q["source_index"], row_sha256=q["row_sha256"],
                    reasons=q["hard_errors"], source="hard_error") for q in report["queries"] if q["hard_errors"]}
    if reviews is not None:
        if any(reviews.get(k) != report[k] for k in ("schema_version", "category", "split", "dataset_fingerprint")):
            raise ValueError("Review file does not match this audit")
        for kind, key in (("images", "image_id"), ("queries", "source_index")):
            seen = set()
            for row in reviews[kind]:
                ident = row[key]
                if ident in seen:
                    raise ValueError("Duplicate review decision")
                seen.add(ident)
                if row["decision"] not in {"pending", "keep", "exclude", "uncertain"}:
                    raise ValueError("Unknown review decision")
                if kind == "images":
                    expected = report["images"].get(ident)
                    if expected is None or row["sha256"] != expected["sha256"]:
                        raise ValueError("Stale/unknown image review")
                else:
                    if type(ident) is not int or not 0 <= ident < len(raw) or row["row_sha256"] != object_sha(raw[ident]):
                        raise ValueError("Stale/unknown query review")
                if row["decision"] in {"pending", "uncertain"}:
                    continue
                if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("reviewer", "reason")):
                    raise ValueError("A final decision needs reviewer and evidence/reason")
                if row["decision"] == "keep":
                    if ident in (bad_images if kind == "images" else bad_queries):
                        raise ValueError("Restore/re-audit unusable data before keeping it")
                    continue
                entry = dict(source="reviewed", reviewer=row["reviewer"], reasons=[row["reason"]])
                if kind == "images":
                    if ident not in train_ids:
                        raise ValueError("Reviewed image is not a train image")
                    bad_images[ident] = dict(entry, image_id=ident, sha256=row["sha256"])
                else:
                    bad_queries[ident] = dict(entry, source_index=ident, row_sha256=row["row_sha256"])
    for i, row in enumerate(raw):
        if isinstance(row, dict) and any(row.get(k) in bad_images for k in ("candidate", "target")
                                         if isinstance(row.get(k), str)):
            bad_queries.setdefault(i, dict(source_index=i, row_sha256=object_sha(row),
                                          reasons=["incident_to_excluded_image"], source="image_policy"))
    if not 0 <= max_exclusion_fraction < 1:
        raise ValueError("max_exclusion_fraction must be in [0, 1)")
    if not raw or len(bad_queries) / len(raw) > max_exclusion_fraction:
        raise ValueError("Exclusion fraction exceeds safeguard; check dataset path and decisions")
    policy = dict(schema_version=1, category=category, split="train", dataset_fingerprint=fingerprint,
                  excluded_images=list(bad_images.values()), excluded_queries=list(bad_queries.values()),
                  original_queries=len(raw), retained_queries=len(raw)-len(bad_queries))
    # Recheck image bytes and annotation bindings before emitting a usable file.
    validate_policy(root, category, policy)
    return policy


def validate_policy(root, category, policy):
    if not isinstance(policy, dict) or any(not isinstance(policy.get(k), list) for k in ("excluded_images", "excluded_queries")):
        raise ValueError("Expected a compiled exclusion policy, not an audit or review file")
    if policy.get("schema_version") != 1 or policy.get("category") != category or policy.get("split") != "train":
        raise ValueError("Noise policy category/split/schema mismatch")
    if policy.get("dataset_fingerprint") != dataset_fingerprint(root, category):
        raise ValueError("Stale noise policy: dataset changed")
    raw = read_json(Path(root) / "captions" / f"cap.{category}.train.json")
    train_ids = set(read_json(Path(root) / "image_splits" / f"split.{category}.train.json"))
    blocked = set()
    for row in policy["excluded_images"]:
        image_id = row["image_id"]
        if image_id in blocked or image_id not in train_ids:
            raise ValueError("Duplicate/non-train excluded image")
        if file_sha(image_path(root, category, image_id)) != row["sha256"]:
            raise ValueError("Excluded image changed; re-audit before using this policy")
        blocked.add(image_id)
    excluded = set()
    for row in policy["excluded_queries"]:
        i = row["source_index"]
        if type(i) is not int or i in excluded or not 0 <= i < len(raw) or row["row_sha256"] != object_sha(raw[i]):
            raise ValueError("Stale/duplicate/unknown excluded query")
        excluded.add(i)
    incident = {i for i, row in enumerate(raw) if isinstance(row, dict)
                and any(isinstance(row.get(k), str) and row[k] in blocked for k in ("candidate", "target"))}
    if not incident <= excluded or len(excluded) == len(raw):
        raise ValueError("Policy leaves incident queries or removes the whole training set")
    return excluded, blocked


def load_noise_policy(root, category, path=None):
    if not path:
        return set(), set()
    return validate_policy(root, category, read_json(path))
