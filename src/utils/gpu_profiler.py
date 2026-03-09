from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Callable, Iterator
import threading

import torch


def _mb(x: int) -> float:
    return x / (1024 ** 2)


def _env_flag(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value.strip())
    except ValueError:
        return default
    return parsed


class _ProfileState(threading.local):
    def __init__(self) -> None:
        super().__init__()
        self.current_batch: int | None = None
        self.lines_logged: int = 0

_profile_state = _ProfileState()


def is_profile_run_enabled(env_key: str = "PROFILE_RUN") -> bool:
    return _env_flag(env_key)

def profile_one_batch_per_epoch() -> bool:
    return _env_flag("PROFILE_ONE_BATCH_PER_EPOCH")


def profile_every_n_batches(default: int = 1) -> int:
    value = _env_int("PROFILE_EVERY_N_BATCHES", default)
    return max(1, value)


def set_current_profile_batch(batch_idx: int | None) -> None:
    if batch_idx != _profile_state.current_batch:
        _profile_state.current_batch = batch_idx
        _profile_state.lines_logged = 0


def _profile_log_limit() -> int:
    return max(1, _env_int("PROFILE_MAX_CONTEXTS", 1))


def should_log_profile_context() -> bool:
    if _profile_state.current_batch is None:
        return True
    if _profile_state.lines_logged >= _profile_log_limit():
        return False
    return True

def should_profile_batch(batch_idx: int, *, every_n_batches: int | None = None) -> bool:
    if profile_one_batch_per_epoch():
        return batch_idx == 0
    if every_n_batches is None:
        every_n_batches = profile_every_n_batches()
    return batch_idx % max(1, every_n_batches) == 0


def _resolve_device_index(device: torch.device | int | None) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    if isinstance(device, torch.device):
        if device.index is None:
            return torch.cuda.current_device()
        return int(device.index)
    return torch.cuda.current_device()


@contextmanager
def cuda_mem_region(
    name: str,
    *,
    device: torch.device | int | None = None,
    log_fn: Callable[[str], None] | None = None,
    enabled: bool | None = None,
) -> Iterator[None]:
    if enabled is None:
        enabled = is_profile_run_enabled()
    if not enabled or not torch.cuda.is_available():
        yield
        return
    if isinstance(device, torch.device) and device.type != "cuda":
        yield
        return

    if log_fn is None:
        log_fn = lambda msg: print(msg, file=sys.stderr)

    device_index = _resolve_device_index(device)
    torch.cuda.synchronize(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)

    alloc0 = torch.cuda.memory_allocated(device_index)
    res0 = torch.cuda.memory_reserved(device_index)

    try:
        yield
    finally:
        torch.cuda.synchronize(device_index)
        alloc1 = torch.cuda.memory_allocated(device_index)
        res1 = torch.cuda.memory_reserved(device_index)
        peak = torch.cuda.max_memory_allocated(device_index)

        if not should_log_profile_context():
            return
        _profile_state.lines_logged += 1
        log_fn(
            f"[VRAM] {name:20s} "
            f"alloc_delta={_mb(alloc1 - alloc0):7.1f}MB  "
            f"reserved_delta={_mb(res1 - res0):7.1f}MB  "
            f"peak={_mb(peak):7.1f}MB  "
            f"alloc={_mb(alloc1):7.1f}MB  "
            f"reserved={_mb(res1):7.1f}MB"
        )
