from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def atomic_write_json(path: str, payload: Any) -> None:
    ensure_dir(os.path.dirname(path))
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def build_run_dir(base_dir: str, run_name: str, timestamp: str) -> str:
    safe_name = run_name.replace(" ", "_")
    return os.path.join(base_dir, f"{safe_name}_{timestamp}")
