"""Official DQU-CIR FashionIQ data construction.

This module is intentionally isolated from ``datasets.py`` and ``newDataset.py``.
It preserves the SIGIR'24 DQU-CIR query semantics while supporting both the
official category-subdirectory image layout and this project's flat image layout.
"""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Callable, Sequence

import cv2
import PIL.Image
import torch


def _draw_text(image, point, text):
    font_scale = 0.7
    thickness = 5
    text_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, baseline = cv2.getTextSize(str(text), font, font_scale, thickness)
    location = (point[0], point[1] + text_size[1])
    cv2.rectangle(
        image,
        (location[0] - 1, location[1] - 2 - baseline),
        (location[0] + text_size[0], location[1] + text_size[1]),
        (255, 255, 255),
        -1,
    )
    cv2.putText(
        image,
        str(text),
        (location[0], location[1] + baseline),
        font,
        font_scale,
        (255, 0, 0),
        text_thickness,
        8,
    )
    return image


def draw_text_line(image, point, text_line: str):
    """Match the text rendering used by the official DQU-CIR repository."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, baseline = cv2.getTextSize(
        str(text_line.split(", ")), font, 0.7, 5
    )
    for index, text in enumerate(text_line.split(", ")):
        if text:
            draw_point = [
                point[0],
                point[1] + (text_size[1] + 2 + baseline) * index,
            ]
            image = _draw_text(image, draw_point, text)
    return image


class FashionIQ(torch.utils.data.Dataset):
    """FashionIQ Dress/Shirt/Toptee dataset with official DQU-CIR inputs."""

    def __init__(
        self,
        path: str,
        category: str,
        transform: Sequence[Callable],
        split: str = "original-split",
    ) -> None:
        super().__init__()
        if split not in {"original-split", "val-split"}:
            raise ValueError(f"Unsupported FashionIQ split: {split}")
        self.path = Path(path)
        self.category = category
        self.image_dir = self.path / "resized_image"
        self.split_dir = self.path / "image_splits"
        self.caption_dir = self.path / "captions"
        self.transform = transform
        self.split = split

        self.correction_dict = self._load_json(
            self.caption_dir / f"correction_dict_{category}.json"
        )
        self.train_captions = self._load_json(
            self.caption_dir / f"image_captions_{category}_train.json"
        )
        self.key_words = self._load_json(
            self.caption_dir / f"keywords_in_mods_{category}.json"
        )
        self.train_data = self._build_train_data()
        self.test_queries, self.test_targets = self._build_test_data()

    @staticmethod
    def _load_json(path: Path):
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _correct_text(self, text: str) -> str:
        translation = str.maketrans({key: " " for key in string.punctuation})
        tokens = str(text).lower().translate(translation).strip().split()
        return " ".join(self.correction_dict.get(token, token) for token in tokens)

    def _concat_text(self, captions: Sequence[str]) -> str:
        return "{} and {}".format(
            self._correct_text(captions[0]),
            self._correct_text(captions[1]),
        )

    def _build_train_data(self):
        triplets = self._load_json(
            self.caption_dir / f"cap.{self.category}.train.json"
        )
        return [
            {
                "target": item["target"],
                "candidate": item["candidate"],
                "captions": self._concat_text(item["captions"]),
            }
            for item in triplets
        ]

    def __len__(self) -> int:
        return len(self.train_data)

    def __getitem__(self, index: int):
        item = self.train_data[index]
        candidate = item["candidate"]
        target = item["target"]
        modification = item["captions"]
        target_image, target_path = self.get_img(target, stage=0)
        visual_query, source_path = self.get_written_img(candidate, target, stage=0)
        return {
            "target_img_data": target_image,
            "target_img_path": target_path,
            "source_img_path": source_path,
            "mod": {"str": modification},
            "textual_query": (
                self.train_captions[candidate] + ", but " + modification
            ),
            "visual_query": visual_query,
        }

    def _image_path(self, image_name: str) -> Path:
        flat = self.image_dir / f"{image_name}.jpg"
        if flat.exists():
            return flat
        nested = self.image_dir / self.category / f"{image_name}.jpg"
        if nested.exists():
            return nested
        raise FileNotFoundError(
            f"FashionIQ image {image_name!r} was not found at {flat} or {nested}"
        )

    def get_img(self, image_name: str, stage: int):
        path = self._image_path(image_name)
        with path.open("rb") as handle:
            image = PIL.Image.open(handle).convert("RGB")
        return self.transform[stage](image), str(path)

    def get_written_img(
        self, candidate: str, target: str, stage: int
    ):
        path = self._image_path(candidate)
        keyword_key = f"{candidate}_{target}"
        if keyword_key not in self.key_words:
            raise KeyError(f"Missing DQU-CIR keyword entry: {keyword_key}")
        keyword = self.key_words[keyword_key][-1]
        candidate_image = cv2.imread(str(path))
        if candidate_image is None:
            raise OSError(f"OpenCV could not read {path}")
        candidate_image = cv2.resize(candidate_image, (512, 512))
        written_image = draw_text_line(candidate_image, (15, 15), keyword)
        written_image = PIL.Image.fromarray(
            cv2.cvtColor(written_image, cv2.COLOR_BGR2RGB)
        )
        return self.transform[stage](written_image), str(path)

    def _build_test_data(self):
        images = self._load_json(
            self.split_dir / f"split.{self.category}.val.json"
        )
        image_to_id = {image_name: index for index, image_name in enumerate(images)}
        triplets = self._load_json(
            self.caption_dir / f"cap.{self.category}.val.json"
        )
        image_captions = self._load_json(
            self.caption_dir / f"image_captions_{self.category}_val.json"
        )

        queries = []
        for item in triplets:
            candidate = item["candidate"]
            target = item["target"]
            modification = self._concat_text(item["captions"])
            visual_query, source_path = self.get_written_img(
                candidate, target, stage=1
            )
            queries.append(
                {
                    "visual_query": visual_query,
                    "source_img_path": source_path,
                    "source_img_id": image_to_id[candidate],
                    "textual_query": (
                        image_captions[candidate] + ", but " + modification
                    ),
                    "target_img_id": image_to_id[target],
                    "mod": {"str": modification},
                }
            )

        if self.split == "val-split":
            gallery_ids = []
            seen = set()
            for query in queries:
                for image_id in (query["source_img_id"], query["target_img_id"]):
                    if image_id not in seen:
                        gallery_ids.append(image_id)
                        seen.add(image_id)
        else:
            gallery_ids = list(range(len(images)))

        targets = []
        for image_id in gallery_ids:
            image, path = self.get_img(images[image_id], stage=1)
            targets.append(
                {
                    "target_img_id": image_id,
                    "target_img_data": image,
                    "target_img_path": path,
                }
            )
        return queries, targets
