"""Train-only positive labels. No image similarity is treated as ground truth."""

from __future__ import annotations

import argparse
import json
import string
from collections import defaultdict
from pathlib import Path


class PositiveIndex:
    def __init__(self, queries, train_ids, *, category, mode="identity", manifest=None):
        if mode not in {"identity", "annotated", "verified"}:
            raise ValueError(f"Unknown positive mode: {mode}")
        self.queries = {q["source_index"]: q for q in queries}
        if len(self.queries) != len(queries):
            raise ValueError("Duplicate source_index")
        self.positives = {i: {q["target"]} for i, q in self.queries.items()}
        train_ids = set(train_ids)
        for q in queries:
            if q["candidate"] not in train_ids or q["target"] not in train_ids:
                raise ValueError("Training query contains an image outside the train split")
        if mode in {"annotated", "verified"}:
            groups = defaultdict(set)
            for q in queries:
                groups[(q["candidate"], q["modification_text"])].add(q["target"])
            for i, q in self.queries.items():
                self.positives[i].update(groups[(q["candidate"], q["modification_text"])])
        if mode == "verified":
            if manifest is None:
                raise ValueError("verified mode requires a reviewed manifest")
            if manifest.get("split") != "train" or manifest.get("category") != category:
                raise ValueError("Positive manifest category/split mismatch")
            if manifest.get("schema_version") != 1:
                raise ValueError("Expected manifest schema_version=1")
            seen = set()
            for row in manifest["queries"]:
                i = row["source_index"]
                if i in seen or i not in self.queries:
                    raise ValueError(f"Duplicate or unknown query index: {i}")
                seen.add(i)
                q = self.queries[i]
                # Text and both IDs bind the review to this exact dataset row.
                for key in ("candidate", "target", "modification_text"):
                    if row[key] != q[key]:
                        raise ValueError(f"Stale manifest at query {i}: {key}")
                if row.get("reviewed") is not True:
                    raise ValueError(f"Query {i} has not been reviewed")
                targets = row["positive_targets"]
                if not isinstance(targets, list) or any(not isinstance(t, str) for t in targets):
                    raise ValueError("positive_targets must be a list of image IDs")
                for target in targets:
                    if target not in train_ids or target == q["candidate"]:
                        raise ValueError(f"Invalid/non-train positive at query {i}: {target}")
                self.positives[i].update(targets)
        for i, q in self.queries.items():
            if q["candidate"] in self.positives[i]:
                raise ValueError(f"Reference image cannot be a positive: query {i}")

    def extras(self, index):
        return sorted(self.positives[index] - {self.queries[index]["target"]})

    def mask_rows(self, indices, target_ids):
        return [[target in self.positives[int(i)] for target in target_ids] for i in indices]

    def summary(self):
        counts = [len(p) for p in self.positives.values()]
        return {"queries": len(counts), "multi_positive_queries": sum(c > 1 for c in counts),
                "positive_links": sum(counts)}


def main():
    """Audit annotations without loading CLIP, images, or CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--queries",
                        help="JSON list of normalized dataset.train_data records")
    source.add_argument("--fashioniq-root", help="FashionIQ root; read raw train captions with DQU corrections")
    parser.add_argument("--train-split", help="Required with --queries")
    parser.add_argument("--category", required=True, choices=("dress", "shirt", "toptee"))
    parser.add_argument("--mode", default="annotated", choices=("identity", "annotated", "verified"))
    parser.add_argument("--manifest")
    parser.add_argument("--review-template", help="Write an UNREVIEWED template; never auto-label new positives")
    args = parser.parse_args()
    if args.manifest and args.mode != "verified":
        parser.error("--manifest requires --mode verified")
    read = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    if args.fashioniq_root:
        root = Path(args.fashioniq_root)
        captions = root / "captions"
        corrections_path = captions / f"correction_dict_{args.category}.json"
        corrections = read(corrections_path) if corrections_path.exists() else {}
        translation = str.maketrans({key: " " for key in string.punctuation})
        def correct(text):
            return " ".join(corrections.get(t, t) for t in str(text).lower().translate(translation).split())
        queries = [{"source_index": i, "candidate": q["candidate"], "target": q["target"],
                    "modification_text": " and ".join(correct(t) for t in q["captions"])}
                   for i, q in enumerate(read(captions / f"cap.{args.category}.train.json"))]
        train_ids = read(root / "image_splits" / f"split.{args.category}.train.json")
    else:
        if not args.train_split:
            parser.error("--queries requires --train-split")
        queries, train_ids = read(args.queries), read(args.train_split)
    index = PositiveIndex(queries, train_ids, category=args.category,
                          mode=args.mode, manifest=read(args.manifest) if args.manifest else None)
    print(json.dumps(index.summary(), indent=2))
    if args.review_template:
        template = {"schema_version": 1, "category": args.category, "split": "train",
                    "queries": [{**q, "reviewed": False, "positive_targets": []} for q in queries]}
        with Path(args.review_template).open("x", encoding="utf-8") as handle:
            json.dump(template, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
