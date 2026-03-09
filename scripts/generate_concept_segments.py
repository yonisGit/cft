from __future__ import annotations

import argparse
import json
import os
import random
import sys
from contextlib import nullcontext, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src.concept.concept_bank import (
    default_concept_bank_path,
    get_concept_bank_hash,
    get_max_concept_weight,
    get_weighted_concepts,
)
from src.concept.dataset_utils import (
    Sample,
    build_dataset_for_split,
    collect_samples,
    resolve_dataset_splits,
)
from src.utils.logging import get_logger, setup_logging

try:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
except ImportError as exc:
    raise RuntimeError("transformers is required for GroundingDINO.") from exc

logger = get_logger(__name__)

DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.65

DEFAULT_OUT_ROOT = Path("data/datasets_segmentations")

MIN_FG_FRACTION = 0.001
MAX_FG_FRACTION = 0.98


@dataclass(frozen=True)
class Sam2Info:
    config: str
    checkpoint: str


@dataclass
class SampleContext:
    rel_path: str
    label_id: int
    class_name: str
    out_path: Path
    meta_path: Path
    concept_masks_path: Path
    image: Image.Image
    image_np: np.ndarray
    height: int
    width: int
    max_weighted_mask: np.ndarray
    concept_records: list[dict[str, Any]] = field(default_factory=list)
    concept_masks: list[np.ndarray] = field(default_factory=list)


class SampleDataset(Dataset):
    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        return {
            "image": image,
            "label_id": sample.label_id,
            "class_name": sample.class_name,
            "rel_path": sample.rel_path,
            "path": sample.path,
        }


def _require_gsam_env() -> None:
    env_name = os.getenv("CONDA_DEFAULT_ENV")
    if env_name != "gsam_local":
        raise RuntimeError(
            f"GroundedSAM masking must run inside conda env 'gsam_local'. Detected CONDA_DEFAULT_ENV={env_name!r}."
        )


def _default_dataset_root() -> Path:
    env_root = os.getenv("CFT_DATASET_ROOT") or os.getenv("CFT_DATA_ROOT")
    return Path(env_root) if env_root else Path("data")


def _resolve_out_root(out_root: Optional[Path], dataset_name: str, bank_hash: str) -> Path:
    base = out_root or DEFAULT_OUT_ROOT
    if base.name != dataset_name and not base.name.startswith("bank_"):
        base = base / dataset_name

    bank_dir = f"bank_{bank_hash}"
    if base.name != bank_dir:
        base = base / bank_dir

    return base


def _build_prompt(concept: str) -> str:
    prompt = concept.strip()
    return prompt if prompt.endswith(".") else f"{prompt}."


def _collate_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


def _resolve_sam2_checkpoint(arg: Optional[str]) -> Optional[str]:
    if arg:
        if os.path.isfile(arg):
            return arg
        if os.path.isdir(arg):
            candidates = [
                "sam2.1_hiera_large.pt", "sam2.1_hiera_l.pt",
                "sam2_hiera_large.pt", "sam2_hiera_l.pt",
                "sam2_hiera_large.pth", "sam2_hiera_l.pth",
            ]
            for name in candidates:
                path = os.path.join(arg, name)
                if os.path.isfile(path):
                    return path

    env_keys = ["SAM2_CHECKPOINT", "SAM2_CKPT", "SAM2_WEIGHTS", "SAM2_CKPT_PATH"]
    for key in env_keys:
        val = os.getenv(key)
        if val and os.path.isfile(val):
            return val
        if val and os.path.isdir(val):
            if maybe_path := _resolve_sam2_checkpoint(val):
                return maybe_path

    return None


def _normalize_sam2_config(value: str) -> str:
    if os.path.isfile(value):
        with suppress(Exception):
            import importlib.resources as resources
            base = resources.files("sam2")
            return Path(value).relative_to(Path(base)).as_posix()

        norm_value = value.replace("\\", "/")
        if "/configs/" in norm_value:
            return "configs/" + norm_value.split("/configs/", 1)[1]

    return value


def _resolve_sam2_config(arg: Optional[str]) -> Optional[str]:
    if arg:
        return _normalize_sam2_config(arg)

    if env_val := os.getenv("SAM2_CONFIG"):
        return _normalize_sam2_config(env_val)

    with suppress(Exception):
        import importlib.resources as resources
        base = resources.files("sam2")
        cfg_candidates = [
            "sam2.1_hiera_l.yaml", "sam2.1_hiera_b+.yaml", "sam2.1_hiera_s.yaml", "sam2.1_hiera_t.yaml",
            "sam2_hiera_l.yaml", "sam2_hiera_b+.yaml", "sam2_hiera_s.yaml", "sam2_hiera_t.yaml",
        ]

        for cfg_dir in ["configs/sam2.1", "configs/sam2"]:
            for name in cfg_candidates:
                path = base / cfg_dir / name
                if path.is_file():
                    return path.relative_to(base).as_posix()

    return None


def _maybe_align_sam2_config(config: str, checkpoint: str) -> str:
    if not checkpoint or not config:
        return config

    ckpt_name = os.path.basename(checkpoint)
    if "sam2.1" not in ckpt_name or "sam2.1" in config:
        return config

    base_name = os.path.basename(config)
    if not base_name.startswith("sam2_hiera_"):
        return config

    candidate = f"configs/sam2.1/{base_name.replace('sam2_hiera_', 'sam2.1_hiera_')}"

    with suppress(Exception):
        import importlib.resources as resources
        if (resources.files("sam2") / candidate).is_file():
            logger.warning(
                "SAM2 config %s does not match sam2.1 checkpoint %s; using %s",
                config, ckpt_name, candidate,
            )
            return candidate

    return config


def _load_sam2_predictor(
        device: torch.device,
        sam2_config: Optional[str],
        sam2_checkpoint: Optional[str],
) -> tuple[object, Sam2Info]:
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as exc:
        raise RuntimeError("sam2 is required. Install sam2 in the gsam_local environment.") from exc

    config = _resolve_sam2_config(sam2_config)
    if not config:
        raise RuntimeError("SAM2 config not found. Provide --sam2-config or set SAM2_CONFIG.")

    checkpoint = _resolve_sam2_checkpoint(sam2_checkpoint)
    if not checkpoint:
        raise RuntimeError("SAM2 checkpoint not found. Provide --sam2-checkpoint or set SAM2_CHECKPOINT.")

    config = _maybe_align_sam2_config(config, checkpoint)

    try:
        model = build_sam2(config, checkpoint, device=device)
    except TypeError:
        model = build_sam2(config, checkpoint)
        if hasattr(model, "to"):
            model = model.to(device)

    model.eval()
    return SAM2ImagePredictor(model), Sam2Info(config=config, checkpoint=checkpoint)


def _predict_sam2_masks(
        predictor: object,
        image_np: np.ndarray,
        boxes_xyxy: np.ndarray,
) -> list[np.ndarray]:
    if len(boxes_xyxy) == 0:
        return []

    if not hasattr(predictor, "set_image"):
        raise RuntimeError("SAM2 predictor missing set_image method.")

    predictor.set_image(image_np)
    masks: list[np.ndarray] = []

    for box in boxes_xyxy:
        box = box.astype(np.float32)
        box_in = box[None, :] if box.ndim == 1 else box

        try:
            pred = predictor.predict(box=box_in, multimask_output=False)
        except TypeError:
            pred = predictor.predict(box=box_in)

        mask = pred[0] if isinstance(pred, tuple) else pred
        mask = np.asarray(mask)

        if mask.ndim == 4:
            mask = mask[0, 0]
        elif mask.ndim == 3:
            mask = mask[0]

        masks.append(mask.astype(np.float32))

    return masks


def _atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp_path, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate concept segmentations with GroundingDINO+SAM2.")
    parser.add_argument("--root", type=Path, default=_default_dataset_root())
    parser.add_argument("--dataset", type=str, default="stanford_dogs")
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test", "full"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--concept-bank", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--sam2-config", type=str, default=None)
    parser.add_argument("--sam2-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--save-concept-masks", action="store_true")
    return parser.parse_args()


def _collect_split_names(args: argparse.Namespace) -> list[str]:
    split_names = resolve_dataset_splits(args.dataset, args.split)
    if "test" in split_names and args.dataset != "stanford_dogs":
        raise ValueError("Split 'test' is only supported for dataset 'stanford_dogs'.")
    return split_names


def _prepare_dino(device: torch.device) -> tuple[Any, torch.nn.Module]:
    processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID)
    model.to(device)
    model.eval()
    return processor, model


def _run_grounding_dino(
        processor: Any,
        model: torch.nn.Module,
        images: list[Image.Image],
        prompt: str,
        target_sizes: list[tuple[int, int]],
        device: torch.device,
        amp_context: Any,
) -> list[dict[str, Any]]:
    prompts = [prompt] * len(images)
    inputs = processor(images=images, text=prompts, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode(), amp_context:
        outputs = model(**inputs)

    post_process_kwargs = {
        "box_threshold": BOX_THRESHOLD,
        "text_threshold": TEXT_THRESHOLD,
        "target_sizes": target_sizes,
    }

    try:
        return processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], **post_process_kwargs
        )
    except TypeError:
        post_process_kwargs["threshold"] = post_process_kwargs.pop("box_threshold")
        return processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], **post_process_kwargs
        )


def _clean_boxes(boxes: np.ndarray, width: int, height: int) -> list[list[float]]:
    cleaned_boxes = []
    for box in boxes:
        x0, y0, x1, y1 = box.tolist()
        x0 = max(0.0, min(x0, width - 1))
        x1 = max(0.0, min(x1, width - 1))
        y0 = max(0.0, min(y0, height - 1))
        y1 = max(0.0, min(y1, height - 1))

        if x1 > x0 and y1 > y0:
            cleaned_boxes.append([x0, y0, x1, y1])

    return cleaned_boxes


def _process_concept_for_context(
        ctx: SampleContext,
        concept: str,
        weight: float,
        concept_mask: np.ndarray,
        save_concept_masks: bool,
) -> tuple[bool, bool]:
    present = bool(np.any(concept_mask > 0))
    mask_fraction = float(np.count_nonzero(concept_mask > 0) / max(1, concept_mask.size))

    ctx.concept_records.append({
        "concept": concept,
        "weight": weight,
        "present": present,
        "mask_fraction": mask_fraction,
        "max_value": float(concept_mask.max()),
    })

    weighted_mask = concept_mask * weight
    ctx.max_weighted_mask = np.maximum(ctx.max_weighted_mask, weighted_mask)

    if save_concept_masks:
        ctx.concept_masks.append(concept_mask.astype(np.float32))

    return True, present


def _evaluate_mask_quality(mask: np.ndarray) -> tuple[float, str]:
    fg_fraction = float(np.count_nonzero(mask > 0) / max(1, mask.size))
    if fg_fraction <= 0:
        return fg_fraction, "empty"
    if fg_fraction < MIN_FG_FRACTION:
        return fg_fraction, "too_small"
    if fg_fraction > MAX_FG_FRACTION:
        return fg_fraction, "too_large"
    return fg_fraction, "ok"


def _save_sample_results(
        ctx: SampleContext,
        dataset_name: str,
        split_name: str,
        concept_bank_path: Path,
        concept_bank_hash: str,
        weighted_concepts: list[tuple[str, float]],
        max_weight: float,
        sam2_info: Sam2Info,
        save_concept_masks: bool,
) -> bool:
    fg_fraction, quality_status = _evaluate_mask_quality(ctx.max_weighted_mask)

    if quality_status != "ok":
        logger.warning(
            "Mask quality issue (%s) split=%s rel=%s fg_fraction=%.4f",
            quality_status, split_name, ctx.rel_path, fg_fraction,
        )

    meta = {
        "dataset": dataset_name,
        "split": split_name,
        "label_id": ctx.label_id,
        "class_name": ctx.class_name or f"class_{ctx.label_id}",
        "image_relative_path": ctx.rel_path,
        "concept_bank": {
            "path": str(concept_bank_path),
            "hash": concept_bank_hash,
            "num_concepts": len(weighted_concepts),
            "max_weight": max_weight,
        },
        "concepts": ctx.concept_records,
        "thresholds": {
            "box_threshold": BOX_THRESHOLD,
            "text_threshold": TEXT_THRESHOLD,
        },
        "model_ids": {
            "grounding_dino": DINO_MODEL_ID,
            "sam2_config": sam2_info.config,
            "sam2_checkpoint": sam2_info.checkpoint,
        },
        "image_size": {"height": ctx.height, "width": ctx.width},
        "mask_quality": {
            "fg_fraction": fg_fraction,
            "min_fg_fraction": MIN_FG_FRACTION,
            "max_fg_fraction": MAX_FG_FRACTION,
            "status": quality_status,
        },
    }

    _atomic_write_npz(ctx.out_path, {"W_out": ctx.max_weighted_mask.astype(np.float32)})
    _atomic_write_json(ctx.meta_path, meta)

    if save_concept_masks:
        concept_masks_arr = np.stack(ctx.concept_masks, axis=0).astype(np.float32)
        _atomic_write_npz(
            ctx.concept_masks_path,
            {
                "concept_masks": concept_masks_arr,
                "concept_names": np.array([c for c, _ in weighted_concepts]),
                "concept_weights": np.array([w for _, w in weighted_concepts], dtype=np.float32),
            },
        )

    return quality_status == "ok"


def _initialize_contexts(
        batch: list[dict[str, Any]],
        out_root: Path,
        split_name: str,
        save_concept_masks: bool,
) -> list[SampleContext]:
    contexts = []
    for item in batch:
        rel_path = item["rel_path"]
        out_path = out_root / split_name / f"{rel_path}.npz"
        meta_path = out_path.with_suffix(".npz.json")
        concept_masks_path = out_path.with_suffix(".concepts.npz")

        if out_path.exists() and meta_path.exists():
            if not save_concept_masks or concept_masks_path.exists():
                continue

        image_np = np.array(item["image"])
        height, width = image_np.shape[:2]

        contexts.append(
            SampleContext(
                rel_path=rel_path,
                label_id=int(item["label_id"]),
                class_name=item["class_name"],
                out_path=out_path,
                meta_path=meta_path,
                concept_masks_path=concept_masks_path,
                image=item["image"],
                image_np=image_np,
                height=height,
                width=width,
                max_weighted_mask=np.zeros((height, width), dtype=np.float32),
            )
        )
    return contexts


def _run_split(
        dataset_name: str,
        split_name: str,
        ds: Dataset,
        weighted_concepts: list[tuple[str, float]],
        concept_bank_path: Path,
        concept_bank_hash: str,
        max_weight: float,
        processor: Any,
        dino_model: torch.nn.Module,
        sam2_predictor: Any,
        sam2_info: Sam2Info,
        out_root: Path,
        seed: Optional[int],
        limit: Optional[int],
        num_workers: int,
        batch_size: int,
        device: torch.device,
        save_concept_masks: bool,
) -> None:
    samples = collect_samples(dataset_name, ds, split_name=split_name, seed=seed, limit=limit)
    if not samples:
        logger.warning("No samples found for dataset=%s split=%s", dataset_name, split_name)
        return

    loader = DataLoader(
        SampleDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        collate_fn=_collate_batch,
    )

    skipped = failures = processed = 0
    concept_hits = {concept: 0 for concept, _ in weighted_concepts}
    concept_total = {concept: 0 for concept, _ in weighted_concepts}
    amp_context = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()

    logger.info(
        "Processing dataset=%s split=%s samples=%d out_root=%s",
        dataset_name, split_name, len(samples), out_root,
    )

    for batch in tqdm(loader, desc=f"{split_name}"):
        contexts = _initialize_contexts(batch, out_root, split_name, save_concept_masks)

        if not contexts:
            skipped += len(batch)
            continue

        images = [ctx.image for ctx in contexts]
        target_sizes = [(ctx.height, ctx.width) for ctx in contexts]

        for concept, weight in weighted_concepts:
            prompt = _build_prompt(concept)
            dino_results = _run_grounding_dino(
                processor, dino_model, images, prompt, target_sizes, device, amp_context
            )

            for ctx, result in zip(contexts, dino_results):
                boxes = result["boxes"].detach().cpu().numpy() if len(result["boxes"]) else np.zeros((0, 4))
                cleaned_boxes = _clean_boxes(boxes, ctx.width, ctx.height)

                if cleaned_boxes:
                    with torch.inference_mode(), amp_context:
                        masks = _predict_sam2_masks(
                            sam2_predictor,
                            image_np=ctx.image_np,
                            boxes_xyxy=np.array(cleaned_boxes),
                        )
                    concept_mask = np.maximum.reduce(masks) if masks else np.zeros((ctx.height, ctx.width),
                                                                                   dtype=np.float32)
                else:
                    concept_mask = np.zeros((ctx.height, ctx.width), dtype=np.float32)

                attempted, present = _process_concept_for_context(ctx, concept, weight, concept_mask,
                                                                  save_concept_masks)
                if attempted:
                    concept_total[concept] += 1
                if present:
                    concept_hits[concept] += 1

        for ctx in contexts:
            success = _save_sample_results(
                ctx, dataset_name, split_name, concept_bank_path, concept_bank_hash,
                weighted_concepts, max_weight, sam2_info, save_concept_masks,
            )
            if success:
                processed += 1
            else:
                failures += 1

    logger.info("Done dataset=%s split=%s skipped=%d mask_quality_issues=%d", dataset_name, split_name, skipped,
                failures)

    if processed:
        for concept, weight in weighted_concepts:
            total, hits = concept_total.get(concept, 0), concept_hits.get(concept, 0)
            rate = (hits / total * 100.0) if total else 0.0
            logger.info(
                "Concept success dataset=%s split=%s concept=%s weight=%.3f hits=%d/%d (%.1f%%)",
                dataset_name, split_name, concept, weight, hits, total, rate,
            )


def main() -> int:
    setup_logging()
    _require_gsam_env()
    args = _parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("CUDA is required and must have at least 1 device available.")
    elif not args.prepare_only:
        raise RuntimeError("CPU is only supported with --prepare-only.")

    dataset_name = args.dataset
    if not args.prepare_only:
        concept_bank_path = args.concept_bank or default_concept_bank_path()
        if not concept_bank_path.exists():
            raise FileNotFoundError(f"Concept bank not found: {concept_bank_path}")

        weighted_concepts = get_weighted_concepts(dataset_name, concept_bank_path=concept_bank_path)
        concept_bank_hash = get_concept_bank_hash(concept_bank_path)
        max_weight = get_max_concept_weight(weighted_concepts)
        out_root = _resolve_out_root(args.out_root, dataset_name, concept_bank_hash)

    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be > 0, got {args.batch_size}")

    device = torch.device(args.device)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    processor, dino_model = _prepare_dino(device)
    sam2_predictor, sam2_info = _load_sam2_predictor(
        device=device,
        sam2_config=args.sam2_config,
        sam2_checkpoint=args.sam2_checkpoint,
    )

    if args.prepare_only:
        logger.info("Model preparation complete; exiting as requested.")
        return 0

    for split_name in _collect_split_names(args):
        ds = build_dataset_for_split(dataset_name, root=str(args.root), split=split_name)
        _run_split(
            dataset_name=dataset_name,
            split_name=split_name,
            ds=ds,
            weighted_concepts=weighted_concepts,
            concept_bank_path=concept_bank_path,
            concept_bank_hash=concept_bank_hash,
            max_weight=max_weight,
            processor=processor,
            dino_model=dino_model,
            sam2_predictor=sam2_predictor,
            sam2_info=sam2_info,
            out_root=out_root,
            seed=args.seed,
            limit=args.limit,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            device=device,
            save_concept_masks=args.save_concept_masks,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())