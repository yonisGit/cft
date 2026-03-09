from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

from config import constants
from data.registry import build_dataset
from utils.logging import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class Sample:
    path: str
    label_id: int
    class_name: str
    rel_path: str


DATASET_DEFAULT_SPLITS: dict[str, list[str]] = {
    "stanford_dogs": ["train", "val", "test"],
    "cub_birds": ["train", "val"],
    "waterbirds": ["train", "val"],
    "imagenet": ["train", "val"],
    "celeba_masks": ["train", "val"],
    "coco_single": ["train", "val"],
    "voc_single": ["train", "val"],
}


def resolve_dataset_splits(
    dataset_name: str, split: Optional[str]
) -> list[str]:
    if split is not None:
        return [split]
    return DATASET_DEFAULT_SPLITS.get(dataset_name, ["train", "val"])


def build_dataset_for_split(
    dataset_name: str, *, root: str, split: str
):
    actual_split = split
    if dataset_name == "stanford_dogs" and split in ("train", "val", "test"):
        actual_split = "full"
    return build_dataset(
        dataset_name,
        root=root,
        split=actual_split,
        transform=None,
        download=False,
    )


def _normalize_relpath(path: str) -> str:
    rel = path.replace("\\", "/")
    return rel.lstrip("/")


def _class_name_from_dataset(ds, label_id: int) -> str:
    classes = getattr(ds, "classes", None)
    if isinstance(classes, (list, tuple)) and 0 <= label_id < len(classes):
        return str(classes[label_id])
    return f"class_{label_id}"


def _apply_seed_limit(
    samples: list[Sample],
    *,
    seed: Optional[int],
    limit: Optional[int],
) -> list[Sample]:
    if not samples:
        return samples
    samples = sorted(samples, key=lambda s: s.rel_path)
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(samples)
    if limit is not None and limit >= 0:
        samples = samples[:limit]
    return samples


def _safe_stanford_split_paths(ds, split_name: str, path: str) -> Optional[set[str]]:
    if not os.path.exists(path):
        logger.warning(
            "Split %s missing at %s; skipping split.", split_name, path
        )
        return None
    try:
        return ds._load_stanford_extra_split_paths(path)
    except Exception as exc:
        logger.warning("Failed to load split %s (%s): %s", split_name, path, exc)
        return None


def _collect_stanford_dogs_samples(
    ds,
    *,
    split_name: str,
    seed: Optional[int],
    limit: Optional[int],
) -> list[Sample]:
    if split_name == "train":
        split_paths = _safe_stanford_split_paths(
            ds, split_name, constants.STANFORD_DOGS_EXTRA_TRAIN_SPLIT_PATH
        )
    elif split_name == "val":
        split_paths = _safe_stanford_split_paths(
            ds, split_name, constants.STANFORD_DOGS_EXTRA_VAL_SPLIT_PATH
        )
    elif split_name == "test":
        split_paths = _safe_stanford_split_paths(
            ds, split_name, constants.STANFORD_DOGS_EXTRA_TEST_SPLIT_PATH
        )
    elif split_name == "full":
        split_paths = set()
    else:
        raise ValueError(f"Unknown Stanford Dogs split {split_name!r}.")

    if split_paths is None:
        return []

    samples: list[Sample] = []
    for path, target in ds.samples:
        rel = ds._relpath_from_sample_path(path)
        if split_paths and rel not in split_paths:
            continue
        label_id = int(target)
        class_name = _class_name_from_dataset(ds, label_id)
        samples.append(
            Sample(
                path=path,
                label_id=label_id,
                class_name=class_name,
                rel_path=_normalize_relpath(rel),
            )
        )
    return _apply_seed_limit(samples, seed=seed, limit=limit)


def _collect_imagefolder_samples(
    ds,
    *,
    seed: Optional[int],
    limit: Optional[int],
) -> list[Sample]:
    base_root = getattr(ds, "root", None) or ""
    samples: list[Sample] = []
    for path, target in ds.samples:
        label_id = int(target)
        rel = os.path.relpath(path, base_root) if base_root else os.path.basename(path)
        class_name = _class_name_from_dataset(ds, label_id)
        samples.append(
            Sample(
                path=path,
                label_id=label_id,
                class_name=class_name,
                rel_path=_normalize_relpath(rel),
            )
        )
    return _apply_seed_limit(samples, seed=seed, limit=limit)


def _collect_celeb_samples(
    ds,
    *,
    seed: Optional[int],
    limit: Optional[int],
) -> list[Sample]:
    samples: list[Sample] = []
    for idx, entry in enumerate(ds.samples):
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            continue
        path = entry[0]
        img_id = entry[2]
        label_id = int(ds.labels[idx])
        class_name = _class_name_from_dataset(ds, label_id)
        samples.append(
            Sample(
                path=path,
                label_id=label_id,
                class_name=class_name,
                rel_path=_normalize_relpath(str(img_id)),
            )
        )
    return _apply_seed_limit(samples, seed=seed, limit=limit)


def _collect_imagenet_samples(
    ds,
    *,
    seed: Optional[int],
    limit: Optional[int],
) -> list[Sample]:
    samples: list[Sample] = []
    for idx, entry in enumerate(ds.samples):
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            continue
        img_path = entry[0]
        rel_id = entry[2]
        label_id = int(ds.labels[idx])
        if label_id < 0:
            logger.warning(
                "Skipping ImageNet sample with unknown label: %s", rel_id
            )
            continue
        class_name = _class_name_from_dataset(ds, label_id)
        rel_path = f"{rel_id}.JPEG"
        samples.append(
            Sample(
                path=img_path,
                label_id=label_id,
                class_name=class_name,
                rel_path=_normalize_relpath(rel_path),
            )
        )
    return _apply_seed_limit(samples, seed=seed, limit=limit)


def _collect_dict_samples(
    ds,
    *,
    seed: Optional[int],
    limit: Optional[int],
) -> list[Sample]:
    samples: list[Sample] = []
    for entry in ds.samples:
        if not isinstance(entry, dict):
            continue
        img_path = entry.get("img_path")
        label_id = entry.get("class_idx")
        if img_path is None or label_id is None:
            continue
        label_id = int(label_id)
        class_name = entry.get("class_name") or _class_name_from_dataset(
            ds, label_id
        )
        if "file_name" in entry:
            rel = str(entry["file_name"])
        elif "image_id" in entry:
            rel = os.path.basename(str(img_path))
        else:
            rel = os.path.basename(str(img_path))
        samples.append(
            Sample(
                path=str(img_path),
                label_id=label_id,
                class_name=str(class_name),
                rel_path=_normalize_relpath(rel),
            )
        )
    return _apply_seed_limit(samples, seed=seed, limit=limit)


def _collect_path_list_samples(
    ds,
    *,
    seed: Optional[int],
    limit: Optional[int],
) -> list[Sample]:
    samples: list[Sample] = []
    img_paths = getattr(ds, "img_paths", None)
    rel_paths = getattr(ds, "rel_paths", None)
    labels = getattr(ds, "labels", None)
    if not img_paths or not rel_paths or labels is None:
        return samples
    for path, rel, label_id in zip(img_paths, rel_paths, labels):
        label_id = int(label_id)
        class_name = _class_name_from_dataset(ds, label_id)
        samples.append(
            Sample(
                path=str(path),
                label_id=label_id,
                class_name=class_name,
                rel_path=_normalize_relpath(str(rel)),
            )
        )
    return _apply_seed_limit(samples, seed=seed, limit=limit)


def collect_samples(
    dataset_name: str,
    ds,
    *,
    split_name: str,
    seed: Optional[int],
    limit: Optional[int],
) -> list[Sample]:
    if dataset_name == "stanford_dogs":
        return _collect_stanford_dogs_samples(
            ds, split_name=split_name, seed=seed, limit=limit
        )

    if hasattr(ds, "img_paths") and hasattr(ds, "rel_paths"):
        samples = _collect_path_list_samples(ds, seed=seed, limit=limit)
        if samples:
            return samples

    if hasattr(ds, "samples") and ds.samples:
        sample0 = ds.samples[0]
        if isinstance(sample0, (list, tuple)):
            if len(sample0) == 2:
                return _collect_imagefolder_samples(ds, seed=seed, limit=limit)
            if len(sample0) == 3:
                return _collect_celeb_samples(ds, seed=seed, limit=limit)
            if len(sample0) >= 4:
                return _collect_imagenet_samples(ds, seed=seed, limit=limit)
        if isinstance(sample0, dict):
            return _collect_dict_samples(ds, seed=seed, limit=limit)

    raise RuntimeError(
        f"Unsupported dataset adapter for {dataset_name!r} (type={type(ds)})."
    )
