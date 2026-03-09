from __future__ import annotations

import platform
import subprocess
from typing import Any

import torch


def _safe_cmd(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True)
    except Exception:
        return None
    return result.stdout.strip() or None


def collect_run_metadata() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "git_commit": _safe_cmd(["git", "rev-parse", "HEAD"]),
        "git_branch": _safe_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }
