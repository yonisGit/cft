#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from PIL import ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from concept.concept_bank import (
    default_concept_bank_path,
    get_concept_bank_hash,
    get_max_concept_weight,
    get_weighted_concepts,
)
from concept.dataset_utils import build_dataset_for_split, collect_samples
from utils.logging import get_logger, setup_logging


logger = get_logger(__name__)

DEFAULT_OUT_ROOT = Path("data/datasets_segmentations")
DEFAULT_SAVE_DIR = ROOT / "plots" / "concepts_segmentations"
DEFAULT_VMIN = None
DEFAULT_VMAX = None
DEFAULT_BIN_SIZE = None
DEFAULT_PALETTE = [
    (0, 34, 78),
    (8, 51, 112),
    (53, 69, 108),
    (79, 87, 108),
    (102, 105, 112),
    (125, 124, 120),
    (148, 142, 119),
    (174, 163, 113),
    (200, 184, 102),
    (229, 207, 82),
    (254, 232, 56),
]


def _default_dataset_root() -> Path:
    return Path(
        os.getenv("CFT_DATASET_ROOT")
        or os.getenv("CFT_DATA_ROOT")
        or "data"
    )


def _resolve_mask_root(
    mask_root: Optional[Path], dataset_name: str, bank_hash: str
) -> Path:
    base = mask_root or DEFAULT_OUT_ROOT
    if mask_root is not None and base.name.startswith("bank_"):
        return base
    if base.name == dataset_name or base.name.startswith("bank_"):
        base = base
    else:
        base = base / dataset_name
    bank_dir = f"bank_{bank_hash}"
    if base.name != bank_dir:
        base = base / bank_dir
    return base


def _resolve_save_dir(
    save_dir: Optional[Path], dataset_name: str, bank_hash: str
) -> Path:
    base = save_dir or DEFAULT_SAVE_DIR
    if save_dir is not None and base.name.startswith("bank_"):
        return base
    if base.name == dataset_name or base.name.startswith("bank_"):
        base = base
    else:
        base = base / dataset_name
    bank_dir = f"bank_{bank_hash}"
    if base.name != bank_dir:
        base = base / bank_dir
    return base


def _sanitize_name(name: str) -> str:
    safe = name.replace("/", "_").replace(" ", "_")
    return safe.strip("_") or "unknown"


def _load_mask(path: Path) -> np.ndarray:
    data = np.load(path)
    return data["W_out"]


def _load_concept_masks(path: Path) -> Optional[dict[str, np.ndarray]]:
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:
        logger.warning("Failed to read concept masks %s: %s", path, exc)
        return None
    required = {"concept_masks", "concept_names", "concept_weights"}
    if not required.issubset(set(data.files)):
        return None
    return {
        "concept_masks": data["concept_masks"],
        "concept_names": data["concept_names"],
        "concept_weights": data["concept_weights"],
    }


def _load_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to read meta %s: %s", path, exc)
        return {}


def _overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float,
    vmin: float,
    vmax: float,
    bin_size: float,
) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    if vmax <= vmin or bin_size <= 0:
        return overlay.astype(np.uint8)
    bins = np.arange(vmin + bin_size, vmax + 1e-6, bin_size)
    palette = np.array(DEFAULT_PALETTE, dtype=np.float32)
    needed = len(bins) + 1
    if len(palette) < needed:
        logger.warning(
            "Palette has %d colors; need %d bins, repeating last color.",
            len(palette),
            needed,
        )
        last = palette[-1:]
        palette = np.concatenate([palette, np.repeat(last, needed - len(palette) + 1, axis=0)], axis=0)

    bin_idx = np.digitize(mask, bins, right=True) + 1
    bin_idx[mask <= vmin] = 0
    bin_idx = np.clip(bin_idx, 0, len(palette) - 1)
    colors = palette[bin_idx]

    alpha_mask = (alpha * (mask > vmin))[..., None]
    overlay = overlay * (1.0 - alpha_mask) + colors * alpha_mask
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _render_legend(entries: list[tuple[str, tuple[int, int, int]]]) -> Image.Image:
    if not entries:
        entries = [("No weighted concepts detected", (0, 0, 0))]
    padding = 8
    line_height = 16
    swatch = 10
    width = max(
        220,
        max(len(label) for label, _ in entries) * 8 + padding * 3 + swatch,
    )
    height = padding * 2 + line_height * len(entries)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    y = padding
    for label, color in entries:
        x0 = padding
        y0 = y + 2
        draw.rectangle(
            [x0, y0, x0 + swatch, y0 + swatch], fill=color, outline=(0, 0, 0)
        )
        draw.text((padding + swatch + 6, y), label, fill=(0, 0, 0))
        y += line_height
    return canvas


def _color_for_value(
    value: float, *, vmin: float, vmax: float, bin_size: float
) -> tuple[int, int, int]:
    if vmax <= vmin or bin_size <= 0:
        return DEFAULT_PALETTE[0]
    bins = np.arange(vmin + bin_size, vmax + 1e-6, bin_size)
    idx = int(np.digitize([value], bins, right=True)[0]) + 1
    if value <= vmin:
        idx = 0
    idx = max(0, min(idx, len(DEFAULT_PALETTE) - 1))
    return DEFAULT_PALETTE[idx]


def _resolve_range(mask: np.ndarray, vmin: Optional[float], vmax: Optional[float]) -> tuple[float, float]:
    vmin_val = float(mask.min()) if vmin is None else float(vmin)
    vmax_val = float(mask.max()) if vmax is None else float(vmax)
    if vmax_val <= vmin_val:
        vmax_val = vmin_val + 1.0
    return vmin_val, vmax_val


def _resolve_bin_size(vmin: float, vmax: float, bin_size: Optional[float]) -> float:
    if bin_size is not None:
        return float(bin_size)
    span = max(vmax - vmin, 1e-6)
    return span / max(1, len(DEFAULT_PALETTE) - 1)


def _compute_concept_labels(
    concept_data: dict[str, np.ndarray],
    *,
    top_k: int,
) -> list[tuple[str, int, int]]:
    masks = concept_data["concept_masks"]
    names = concept_data["concept_names"]
    weights = concept_data["concept_weights"]

    if masks.ndim != 3 or masks.shape[0] != len(names):
        return []
    masks_bin = masks > 0
    mask_fracs = masks_bin.reshape(masks.shape[0], -1).mean(axis=1)
    scores = mask_fracs * weights
    order = np.argsort(scores)[::-1]

    labels: list[tuple[str, int, int]] = []
    for idx in order[: max(1, top_k)]:
        if mask_fracs[idx] <= 0:
            continue
        ys, xs = np.nonzero(masks_bin[idx])
        if len(ys) == 0:
            continue
        y = int(round(float(ys.mean())))
        x = int(round(float(xs.mean())))
        name = names[idx]
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")
        labels.append((str(name), x, y))
    return labels


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize cached concept masks for a dataset."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_default_dataset_root(),
        help="Dataset root (DATASET_ROOT).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="stanford_dogs",
        help="Dataset name registered in data.registry.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "full"],
        help="Split to visualize.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples to visualize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic sampling.",
    )
    parser.add_argument(
        "--only-cached",
        action="store_true",
        help="Only visualize samples with cached masks.",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=None,
        help="Root directory containing cached masks (defaults to data/datasets_segmentations/<dataset>).",
    )
    parser.add_argument(
        "--concept-bank",
        type=Path,
        default=None,
        help="Path to weighted concept bank JSON (defaults to concept/datasets_concepts_bank.json).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Output directory for overlay images (defaults to plots/concepts_segmentations/<dataset>).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Overlay strength for the weighted mask heatmap.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=DEFAULT_VMIN,
        help="Lower bound for discrete bin scaling (defaults to per-image min).",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=DEFAULT_VMAX,
        help="Upper bound for discrete bin scaling (defaults to per-image max).",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=DEFAULT_BIN_SIZE,
        help="Bin size for discrete color thresholds (defaults to full-range binning).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K concepts to list in the legend.",
    )
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = _parse_args()

    if args.split == "test" and args.dataset != "stanford_dogs":
        raise ValueError(
            "Split 'test' is only supported for dataset 'stanford_dogs'."
        )

    ds = build_dataset_for_split(
        args.dataset, root=str(args.root), split=args.split
    )

    concept_bank_path = (
        args.concept_bank
        if args.concept_bank is not None
        else default_concept_bank_path()
    )
    weighted_concepts = get_weighted_concepts(
        args.dataset, concept_bank_path=concept_bank_path
    )
    max_weight = get_max_concept_weight(weighted_concepts)
    bank_hash = get_concept_bank_hash(concept_bank_path)

    mask_root = _resolve_mask_root(args.mask_root, args.dataset, bank_hash)
    save_dir = _resolve_save_dir(args.save_dir, args.dataset, bank_hash)
    save_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(
        args.dataset,
        ds,
        split_name=args.split,
        seed=args.seed,
        limit=None if args.only_cached else args.num_samples,
    )
    if not samples:
        logger.warning(
            "No samples found for dataset=%s split=%s",
            args.dataset,
            args.split,
        )
        return 0

    if args.only_cached:
        cached_samples = []
        for sample in samples:
            mask_path = mask_root / args.split / f"{sample.rel_path}.npz"
            if mask_path.exists():
                cached_samples.append(sample)
            if len(cached_samples) >= args.num_samples:
                break
        samples = cached_samples
        if not samples:
            logger.warning(
                "No cached masks found for dataset=%s split=%s",
                args.dataset,
                args.split,
            )
            return 0

    for idx, sample in enumerate(samples):
        rel = sample.rel_path
        mask_path = mask_root / args.split / f"{rel}.npz"
        if not mask_path.exists():
            logger.warning("Missing mask for %s", rel)
            continue

        image = Image.open(sample.path).convert("RGB")
        image_np = np.array(image)
        w_out = _load_mask(mask_path)
        meta = _load_meta(mask_path.with_suffix(".npz.json"))
        concept_data = _load_concept_masks(mask_path.with_suffix(".concepts.npz"))
        label_id = int(meta.get("label_id", sample.label_id))
        label_name = meta.get("class_name")
        if not label_name:
            label_name = sample.class_name
        label_name = _sanitize_name(str(label_name))

        if w_out.shape[:2] != image_np.shape[:2]:
            logger.warning("Mask/image size mismatch for %s", rel)
            continue

        vmin, vmax = _resolve_range(w_out, args.vmin, args.vmax)
        bin_size = _resolve_bin_size(vmin, vmax, args.bin_size)
        overlay = _overlay_mask(
            image_np,
            w_out,
            alpha=args.alpha,
            vmin=vmin,
            vmax=vmax,
            bin_size=bin_size,
        )
        labels = []
        if concept_data is not None:
            masks = concept_data["concept_masks"]
            if masks.shape[-2:] == w_out.shape[:2]:
                labels = _compute_concept_labels(concept_data, top_k=args.top_k)

        present = []
        meta_concepts = meta.get("concepts")
        if isinstance(meta_concepts, list):
            for entry in meta_concepts:
                if not isinstance(entry, dict):
                    continue
                mask_fraction = float(entry.get("mask_fraction", 0.0))
                if not entry.get("present") and mask_fraction <= 0.0:
                    continue
                weight = float(entry.get("weight", 0.0))
                score = mask_fraction * weight
                present.append((entry, score, mask_fraction, weight))
        if present:
            present.sort(key=lambda x: x[1], reverse=True)
            entries = []
            for entry, _score, mask_fraction, weight in present[: max(1, args.top_k)]:
                label = f"{entry.get('concept')} (w={weight:.2f}, a={mask_fraction:.3f})"
                color = _color_for_value(
                    weight, vmin=vmin, vmax=vmax, bin_size=bin_size
                )
                entries.append((label, color))
        else:
            entries = [("No weighted concepts detected", (0, 0, 0))]
        legend = _render_legend(entries)

        filename = f"{args.split}_label{label_id}_{label_name}_{idx:03d}"
        out_path = save_dir / f"{filename}_overlay.png"
        legend_path = save_dir / f"{filename}_legend.png"
        overlay_img = Image.fromarray(overlay)
        if labels:
            draw = ImageDraw.Draw(overlay_img)
            for name, x, y in labels:
                x = max(0, min(int(x), overlay_img.width - 1))
                y = max(0, min(int(y), overlay_img.height - 1))
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    draw.text((x + dx, y + dy), name, fill=(0, 0, 0))
                draw.text((x, y), name, fill=(255, 255, 255))
        overlay_img.save(out_path)
        legend.save(legend_path)

    logger.info("Saved overlays to %s", save_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
