from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONCEPT_BANK_PATH = ROOT / "concept" / "datasets_concepts_bank.json"


def default_concept_bank_path() -> Path:
    return DEFAULT_CONCEPT_BANK_PATH


def _load_concept_bank(path: Path) -> dict:
    with path.open("r") as f:
        bank = json.load(f)
    if not isinstance(bank, dict):
        raise ValueError(f"Concept bank must be a dict, got {type(bank)}.")
    return bank


def _validate_weighted_concepts(
    dataset_name: str, concepts: object
) -> list[tuple[str, float]]:
    if not isinstance(concepts, list) or not concepts:
        raise ValueError(
            f"Concept bank entry for {dataset_name} must be a non-empty list."
        )
    weighted: list[tuple[str, float]] = []
    for idx, entry in enumerate(concepts):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(
                f"Concept entry {idx} for {dataset_name} must be [name, weight]."
            )
        name, weight = entry
        if not isinstance(name, str):
            raise ValueError(
                f"Concept name {name!r} for {dataset_name} is not a string."
            )
        if name.lower() != name:
            raise ValueError(
                f"Concept {name!r} for {dataset_name} must be lowercase."
            )
        try:
            weight_val = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Weight {weight!r} for {dataset_name} concept {name!r} is invalid."
            ) from exc
        weighted.append((name, weight_val))
    return weighted


def get_weighted_concepts(
    dataset_name: str, *, concept_bank_path: Optional[Path] = None
) -> list[tuple[str, float]]:
    path = concept_bank_path or DEFAULT_CONCEPT_BANK_PATH
    bank = _load_concept_bank(path)
    if dataset_name not in bank:
        raise KeyError(
            f"Dataset {dataset_name!r} missing from concept bank: {path}"
        )
    return _validate_weighted_concepts(dataset_name, bank[dataset_name])


def get_concept_bank_hash(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def get_max_concept_weight(weighted_concepts: list[tuple[str, float]]) -> float:
    if not weighted_concepts:
        return 0.0
    return max(weight for _, weight in weighted_concepts)
