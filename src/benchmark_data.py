"""Image-free, strict metadata adapters for the DQU CIRR/Shoes protocols."""
from __future__ import annotations

import json
import string
from collections import Counter
from functools import lru_cache
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unique(values, label):
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label}")
    return values


@lru_cache(maxsize=2048)
def _safe_parent(root, parent):
    root = Path(root).resolve()
    resolved = (root / parent).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Image parent escapes dataset: {parent}")
    return resolved


def safe_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Image path escapes dataset: {relative}")
    p = _safe_parent(str(root), str(relative.parent)) / relative.name
    if p.is_symlink() and not p.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f"Image symlink escapes dataset: {relative}")
    return p


class BenchmarkData:
    """Keep query/image IDs stable and do not decode or cache image tensors.

    CIRR supports both the original DQU nested annotation layout and the
    current official flat layout. Images can live in images/, img_raw/, or
    directly under the dataset root. test1 is deliberately not loaded here.
    """
    def __init__(self, root, dataset):
        self.root = Path(root).resolve()
        self.dataset = dataset
        self.image_paths = {}
        self.captions = {}
        self.keywords = {}
        if dataset == "cirr":
            self._cirr()
        elif dataset == "shoes":
            self._shoes()
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        for split, rows in self.queries.items():
            ids = set(self.image_ids[split])
            if not rows or not ids:
                raise ValueError(f"Empty {dataset} {split}")
            for row in rows:
                a, b = row["candidate"], row["target"]
                if (a == b and dataset != "shoes") or a not in ids or b not in ids:
                    raise ValueError(f"Invalid/cross-split {split} query: {row}")
                if not row["modification_text"].strip():
                    raise ValueError("Empty modification text")
                if a not in self.captions[split]:
                    raise ValueError(f"Missing BLIP2 caption for {split}:{a}")
                key = a + "+" + b
                entry = self.keywords[split].get(key)
                if not isinstance(entry, list) or not entry or not isinstance(entry[-1], str):
                    raise ValueError(f"Missing/invalid DQU keywords: {split}:{key}")
                if dataset == "cirr" and split == "val":
                    members = row["subset_id"]
                    unique(members, "CIRR subset IDs")
                    if len(members) != 6 or a not in members or b not in members or not set(members) <= ids:
                        raise ValueError("Invalid CIRR six-image subset")
        if set(self.image_ids["train"]) & set(self.image_ids["val"]):
            raise ValueError("Train/evaluation image leakage")

    def _cirr(self):
        annotation_roots = [self.root, self.root / "captions"]
        matches = [p for p in annotation_roots
                   if (p / "captions/cap.rc2.train.json").is_file()
                   and (p / "image_splits/split.rc2.train.json").is_file()]
        if len(matches) != 1:
            raise ValueError("Need one unambiguous CIRR annotation root (flat or DQU nested layout)")
        annotations = matches[0]
        self.image_ids, self.queries = {}, {}
        for split in ("train", "val"):
            mapping = read_json(annotations / f"image_splits/split.rc2.{split}.json")
            self.image_ids[split] = list(mapping)
            for name, rel in mapping.items():
                candidates = [safe_path(self.root, Path(prefix) / rel)
                              for prefix in ("images", "img_raw", "")]
                found = [p for p in candidates if p.is_file()]
                if len({str(p) for p in found}) > 1:
                    raise ValueError(f"Ambiguous image copies for {name}; choose one image root")
                self.image_paths[name] = found[0] if found else candidates[0]
            self.captions[split] = read_json(self.root / f"image_captions_cirr_{split}.json")
            self.keywords[split] = read_json(self.root / f"keywords_in_mods_cirr_{split}.json")
            self.queries[split] = []
            for i, row in enumerate(read_json(annotations / f"captions/cap.rc2.{split}.json")):
                self.queries[split].append(dict(candidate=row["reference"], target=row["target_hard"],
                    modification_text=row["caption"], source_index=i, pairid=row["pairid"],
                    subset_id=row["img_set"]["members"]))

    def _shoes(self):
        self.image_ids = {split: unique([x.strip() for x in
            (self.root / filename).read_text().splitlines() if x.strip()], f"Shoes {split} IDs")
            for split, filename in (("train", "train_im_names.txt"), ("val", "eval_im_names.txt"))}
        for folder in sorted(self.root.glob("womens*")):
            if folder.is_dir():
                for p in folder.glob("*/*.jpg"):
                    if p.name in self.image_paths:
                        raise ValueError(f"Duplicate Shoes image basename: {p.name}")
                    self.image_paths[p.name] = p
        for ids in self.image_ids.values():
            for name in ids:
                self.image_paths.setdefault(name, safe_path(self.root, Path("MISSING_IMAGES") / name))
        correction = read_json(self.root / "correction_dict_shoes.json")
        translation = str.maketrans({key: " " for key in string.punctuation})
        self.queries = {"train": [], "val": []}
        lookup = {name: split for split, ids in self.image_ids.items() for name in ids}
        # Index the original complete annotation file, not its filtered split.
        for i, row in enumerate(read_json(self.root / "relative_captions_shoes.json")):
            a, b = row["ReferenceImageName"], row["ImageName"]
            if a not in lookup:
                raise ValueError(f"Shoes reference outside both splits: {a}")
            tokens = row["RelativeCaption"].lower().translate(translation).split()
            mod = " ".join(correction.get(token, token) for token in tokens)
            self.queries[lookup[a]].append(dict(candidate=a, target=b, modification_text=mod, source_index=i))
        for split in self.queries:
            self.captions[split] = read_json(self.root / "image_captions_shoes.json")
            self.keywords[split] = read_json(self.root / "keywords_in_mods_shoes.json")

    def baseline_text(self, row, split):
        # CIRR's original implementation concatenates ', but' without a space.
        separator = ", but" if self.dataset == "cirr" else ", but "
        return self.captions[split][row["candidate"]] + separator + row["modification_text"]

    def audit(self):
        missing = [name for name, path in self.image_paths.items() if not path.is_file()]
        target_counts = Counter(row["target"] for row in self.queries["train"])
        return {"dataset": self.dataset, "root": str(self.root),
                "train_queries": len(self.queries["train"]), "eval_queries": len(self.queries["val"]),
                "train_images": len(self.image_ids["train"]), "eval_images": len(self.image_ids["val"]),
                "self_target_train_queries": sum(r["candidate"] == r["target"] for r in self.queries["train"]),
                "self_target_eval_queries": sum(r["candidate"] == r["target"] for r in self.queries["val"]),
                "unique_train_targets": len(target_counts),
                "train_queries_with_reused_target_id": sum(target_counts[row["target"]] > 1 for row in self.queries["train"]),
                "missing_images": len(missing), "missing_examples": missing[:10],
                "ready": not missing, "test1_loaded": False}


def selection_score(metrics, dataset):
    if dataset == "cirr":
        # Match the official DQU training script's sum of all seven recalls.
        keys = [f"cirr_r{k}" for k in (1, 5, 10, 50)] + [f"cirr_subset_r{k}" for k in (1, 2, 3)]
        return sum(metrics[k] for k in keys)
    return metrics[f"{dataset}_r10"] + metrics[f"{dataset}_r50"]


def selection_name(dataset):
    return "sum of 7 CIRR recalls (official DQU selection)" if dataset == "cirr" else "R@10 + R@50"


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("cirr", "shoes"), required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--output")
    args = p.parse_args()
    result = BenchmarkData(args.root, args.dataset).audit()
    content = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(content + "\n", encoding="utf-8")
    print(content)
