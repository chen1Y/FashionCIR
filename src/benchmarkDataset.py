"""Lazy image construction shared by original DQU and structured CIRR/Shoes."""
from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from pathlib import Path
import random

import cv2
from PIL import Image
from torch.utils.data import Dataset

from benchmark_data import BenchmarkData, read_json
from dquDataset import draw_text_line


class LazyRecords(Sequence):
    def __init__(self, records, materialize):
        self.records, self.materialize = records, materialize

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.materialize(row) for row in self.records[index]]
        return self.materialize(self.records[index])


class CIRBenchmark(Dataset):
    def __init__(self, path, category, transform, *, structured_train_path=None,
                 structured_val_path=None, use_written_image=True):
        self.meta = BenchmarkData(path, category)
        audit = self.meta.audit()
        if not audit["ready"]:
            raise FileNotFoundError(f"Missing {audit['missing_images']} images: {audit['missing_examples']}")
        self.path, self.category, self.transform = Path(path), category, transform
        self.use_written_image = use_written_image
        self.noise_excluded_indices, self.noise_excluded_image_ids = set(), set()
        self.train_data = self.meta.queries["train"]
        self.structured = {}
        self._load_structured(structured_train_path, "train")
        self._load_structured(structured_val_path, "val")
        self.train_structured_coverage = len(self.structured["train"]) / len(self.train_data)
        self.val_structured_coverage = len(self.structured["val"]) / len(self.meta.queries["val"])
        self.query_metadata = [dict(row, source_img_id=row["candidate"], target_img_id=row["target"],
                                   **self._structured_values(row, "val"))
                               for row in self.meta.queries["val"]]
        self.target_metadata = [dict(target_img_id=name) for name in self.meta.image_ids["val"]]
        self.test_queries = LazyRecords(self.query_metadata, partial(self._query, split="val"))
        self.test_targets = LazyRecords(self.target_metadata, self._target)

    def _target(self, row):
        return dict(row, target_img_data=self.get_img(row["target_img_id"], 1)[0])

    def _load_structured(self, path, split):
        self.structured[split] = {}
        if path is None:
            return
        # Reuse the slot encoder only; its FashionIQ semantic rules/generator
        # must never be applied to natural scenes or footwear.
        from newDataset import FashionIQ
        by_index = {row["source_index"]: row for row in self.meta.queries[split]}
        payload = read_json(path)
        for sample in payload["samples"]:
            if sample.get("category") != self.category or sample.get("split") != split:
                raise ValueError("Structured dataset/split mismatch")
            idx = sample["query_index"]
            row = by_index.get(idx)
            if row is None or any(sample.get(k) != row[k] for k in ("candidate", "target", "modification_text")):
                raise ValueError("Structured query identity/text mismatch")
            if idx in self.structured[split]:
                raise ValueError("Duplicate structured query index")
            if not (sample.get("status") == "ok" and sample.get("validation", {}).get("valid")
                    and sample.get("semantic_validation", {}).get("valid")):
                continue
            if sample.get("generation_inputs") != "reference_image_and_modification_only":
                raise ValueError("Structured input provenance must declare reference-only generation")
            edit = sample["structured_edit"]
            text = FashionIQ._linearize_structured_edit(edit)
            fields, mask = FashionIQ._linearize_structured_fields(edit)
            fields = tuple(x.replace("Target garment: ", "Target: ") for x in fields)
            confidence = float(edit.get("confidence", {}).get("score", 0))
            complexity = float(edit.get("complexity", {}).get("score", .5))
            if not 0 <= confidence <= 1 or not 0 <= complexity <= 1:
                raise ValueError("Invalid structured confidence/complexity")
            if not text or not any(mask):
                continue
            ambiguity = len(edit.get("confidence", {}).get("ambiguities", []) or [])
            quality = (confidence, 1-complexity, 1/(1+ambiguity),
                       0.0 if sample["semantic_validation"].get("repaired") else 1.0, sum(mask)/len(mask))
            self.structured[split][idx] = (text, fields, mask, True, confidence, quality)

    def __len__(self):
        return len(self.train_data)

    def get_img(self, name, stage=0):
        path = self.meta.image_paths[name]
        with Image.open(path) as image:
            tensor = self.transform[stage](image.convert("RGB"))
        return tensor, str(path)

    def _structured_values(self, row, split):
        text = self.meta.baseline_text(row, split)
        values = self.structured[split].get(row["source_index"],
            (text, (text,)*4, (False,)*4, False, 0.0, (0.0,)*5))
        return dict(zip(("structured_text", "structured_fields", "structured_field_mask",
                         "has_structured_text", "structured_confidence", "structured_quality_features"), values))

    def _query(self, row, split):
        stage = int(split == "val")
        text = self.meta.baseline_text(row, split)
        source_path = str(self.meta.image_paths[row["candidate"]])
        if self.use_written_image:
            image = cv2.imread(source_path)
            if image is None:
                raise FileNotFoundError(source_path)
            # Original Shoes resizes before writing; original CIRR does not.
            if self.category == "shoes":
                image = cv2.resize(image, (512, 512))
            keyword = self.meta.keywords[split][row["candidate"]+"+"+row["target"]][-1]
            image = draw_text_line(image, (15, 15), keyword)
            visual = self.transform[stage](Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        else:
            visual = self.get_img(row["candidate"], stage)[0]
        structured = self.structured[split].get(row["source_index"],
            (text, (text,)*4, (False,)*4, False, 0.0, (0.0,)*5))
        return dict(row, visual_query=visual, textual_query=text, source_img_path=source_path,
            source_img_id=row["candidate"], target_img_id=row["target"], query_index=row["source_index"],
            structured_text=structured[0], structured_fields=structured[1], structured_field_mask=structured[2],
            has_structured_text=structured[3], structured_confidence=structured[4],
            structured_quality_features=structured[5], mod={"str":row["modification_text"]})

    def __getitem__(self, index):
        row = self.train_data[index]
        out = self._query(row, "train")
        target_image, path = self.get_img(row["target"])
        out.update(target_img_data=target_image, target_img_path=path)
        positives = getattr(self, "multi_positive_index", None)
        if positives is not None and self.sample_extra_positive:
            extras = positives.extras(row["source_index"])
            extra = random.choice(extras) if extras else row["target"]
            out.update(extra_target_img=self.get_img(extra)[0] if extras else target_image,
                       extra_target_id=extra, has_extra_target=bool(extras))
        return out
