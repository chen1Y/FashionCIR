"""Replace only the native FashionIQ DQU-CIR textual query with verified Flash text."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from dquDataset import FashionIQ


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_queries(data, texts, category, split):
    spec = read_json(texts / "inventory.json")["splits"][f"{category}_{split}"]
    export = texts / f"{category}_{split}.jsonl"
    annotations = read_json(data / "captions" / f"cap.{category}.{split}.json")
    if spec["path"] != export.name or file_sha256(export) != spec["sha256"]:
        raise ValueError(f"Flash export does not match its manifest: {export}")
    with export.open(encoding="utf-8") as handle:
        generated = [json.loads(line) for line in handle]
    if len(generated) != len(annotations) or len(generated) != spec["rows"]:
        raise ValueError(f"Flash and DQU row counts differ: {export}")

    queries = []
    counts = Counter()
    for index, (row, original) in enumerate(zip(generated, annotations, strict=True)):
        if (
            (row["category"], row["split"], row["source_index"]) != (category, split, index)
            or (row["candidate"], row["target"], row["fragments"])
            != (original["candidate"], original["target"], original["captions"])
        ):
            raise ValueError(f"Flash and DQU query identity differs: {export}:{index + 1}")
        status = row["status"]
        counts[status] += 1
        text = None
        if status == "qwen_used":
            text = row["main_context_text"]
            if row["origin"] != "qwen" or not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid accepted Flash text: {export}:{index + 1}")
        elif status not in {"generation_fallback", "unknown_fallback", "token_budget_fallback"}:
            raise ValueError(f"Unexpected Flash generation status: {export}:{index + 1}: {status}")
        queries.append(text)
    if counts != spec["status_counts"]:
        raise ValueError(f"Flash status counts do not match the manifest: {export}")
    return queries, dict(counts)


class FlashFashionIQ(FashionIQ):
    """Keep native DQU images, targets, metadata, and fallback queries."""

    def __init__(self, path, category, transform, split, noise_manifest=None, *, flash_text_dir):
        if split != "original-split" or noise_manifest is not None:
            raise ValueError("Flash query overrides require the unfiltered original FashionIQ split")
        self.train_text, self.train_text_counts = load_queries(Path(path), flash_text_dir, category, "train")
        self.val_text, self.val_text_counts = load_queries(Path(path), flash_text_dir, category, "val")
        super().__init__(path, category, transform, split, noise_manifest)
        if len(self.train_data) != len(self.train_text) or len(self.test_queries) != len(self.val_text):
            raise ValueError(f"Native DQU dataset changed the Flash source ordering: {category}")

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.train_text[index] is not None:
            item["textual_query"] = self.train_text[index]
        return item

    def _build_test_data(self):
        queries, targets = super()._build_test_data()
        for query, text in zip(queries, self.val_text, strict=True):
            if text is not None:
                query["textual_query"] = text
        return queries, targets
