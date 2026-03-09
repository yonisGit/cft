from __future__ import annotations

import math
import os
import time
from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterable


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
        return int(value.strip())
    except ValueError:
        return default


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if pct <= 0:
        return sorted_vals[0]
    if pct >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


@dataclass(frozen=True)
class WindowSummary:
    count: int
    avg: float
    p50: float
    p95: float
    min: float
    max: float


class WindowStats:
    def __init__(self, window: int) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self._values: deque[float] = deque(maxlen=window)

    def add(self, value: float) -> None:
        self._values.append(float(value))

    def clear(self) -> None:
        self._values.clear()

    def summary(self) -> WindowSummary | None:
        if not self._values:
            return None
        vals = sorted(self._values)
        count = len(vals)
        avg = sum(vals) / float(count)
        return WindowSummary(
            count=count,
            avg=avg,
            p50=_percentile(vals, 50.0),
            p95=_percentile(vals, 95.0),
            min=vals[0],
            max=vals[-1],
        )


class PerfTracker:
    def __init__(
        self,
        *,
        tag: str,
        window: int,
        log_every: int,
        warmup: int,
        sync_cuda: bool,
        log_mem: bool,
    ) -> None:
        if log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {log_every}")
        if warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {warmup}")
        self.tag = tag
        self.window = window
        self.log_every = log_every
        self.warmup = warmup
        self.sync_cuda = sync_cuda
        self.log_mem = log_mem
        self._stats: dict[str, WindowStats] = defaultdict(lambda: WindowStats(window))

    @classmethod
    def from_env(cls, *, tag: str) -> "PerfTracker | None":
        if not _env_flag("CFT_PERF_LOG"):
            return None
        return cls(
            tag=tag,
            window=_env_int("CFT_PERF_WINDOW", 200),
            log_every=_env_int("CFT_PERF_LOG_EVERY", 50),
            warmup=_env_int("CFT_PERF_WARMUP", 10),
            sync_cuda=_env_flag("CFT_PERF_SYNC_CUDA"),
            log_mem=_env_flag("CFT_PERF_LOG_MEM"),
        )

    def reset(self) -> None:
        for stats in self._stats.values():
            stats.clear()

    def record(self, key: str, value: float) -> None:
        self._stats[key].add(value)

    def keys(self) -> list[str]:
        return sorted(self._stats.keys())

    def summary(self) -> dict[str, WindowSummary]:
        out: dict[str, WindowSummary] = {}
        for key, stats in self._stats.items():
            summary = stats.summary()
            if summary is not None:
                out[key] = summary
        return out

    def should_log(self, step: int) -> bool:
        if step < self.warmup:
            return False
        return (step + 1) % self.log_every == 0

    @contextmanager
    def time(self, key: str, *, sync_cuda: bool | None = None) -> Iterable[None]:
        if sync_cuda is None:
            sync_cuda = self.sync_cuda
        if sync_cuda:
            _maybe_cuda_sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            if sync_cuda:
                _maybe_cuda_sync()
            self.record(key, time.perf_counter() - start)

    def log_summary(self, logger, *, step: int) -> None:
        if not self.should_log(step):
            return

        summary = self.summary()
        if not summary:
            return

        main_keys = [k for k in sorted(summary) if not k.startswith("cam.")]
        cam_keys = [k for k in sorted(summary) if k.startswith("cam.")]

        main_line = _format_summary_line(summary, main_keys)
        logger.info("[PERF][%s] step=%s %s", self.tag, step + 1, main_line)

        if cam_keys:
            cam_line = _format_summary_line(summary, cam_keys)
            logger.info("[PERF][%s] step=%s %s", self.tag, step + 1, cam_line)

        if self.log_mem:
            mem_line = _format_mem_line()
            if mem_line:
                logger.info("[PERF][%s] step=%s %s", self.tag, step + 1, mem_line)


def maybe_time(perf: PerfTracker | None, key: str, *, sync_cuda: bool | None = None):
    if perf is None:
        return nullcontext()
    return perf.time(key, sync_cuda=sync_cuda)


def _is_rate_key(key: str) -> bool:
    return key.endswith("_per_s") or key.endswith("_per_sec") or "per_s" in key


def _format_stats(key: str, stats: WindowSummary) -> str:
    if _is_rate_key(key):
        return (
            f"{key}=avg{stats.avg:.1f}/s "
            f"p50{stats.p50:.1f}/s p95{stats.p95:.1f}/s"
        )
    avg_ms = stats.avg * 1000.0
    p50_ms = stats.p50 * 1000.0
    p95_ms = stats.p95 * 1000.0
    return f"{key}=avg{avg_ms:.1f}ms p50{p50_ms:.1f}ms p95{p95_ms:.1f}ms"


def _format_summary_line(
    summary: dict[str, WindowSummary],
    keys: list[str],
) -> str:
    parts = []
    for key in keys:
        stats = summary.get(key)
        if stats is None:
            continue
        parts.append(_format_stats(key, stats))
    return " | ".join(parts)


def _maybe_cuda_sync() -> None:
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()


def _format_mem_line() -> str | None:
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return f"vram_alloc={alloc:.2f}GiB vram_reserved={reserved:.2f}GiB vram_peak={peak:.2f}GiB"


def get_gpu_memory_stats() -> dict[str, float] | None:
    """Get current GPU memory statistics in GiB.
    
    Returns:
        Dictionary with 'allocated', 'reserved', 'peak_allocated' keys,
        or None if CUDA is not available.
    """
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_gib": torch.cuda.memory_allocated() / (1024 ** 3),
        "reserved_gib": torch.cuda.memory_reserved() / (1024 ** 3),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
    }


def reset_gpu_peak_memory() -> None:
    """Reset GPU peak memory statistics for fresh tracking."""
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class EpochPerfTracker:
    """Tracks performance metrics across an entire epoch.
    
    Usage:
        tracker = EpochPerfTracker(tag="train", device=device)
        for batch in loader:
            with tracker.step():
                # training step
        summary = tracker.epoch_summary()
    """
    
    def __init__(self, *, tag: str, device=None) -> None:
        self.tag = tag
        self.device = device
        self._step_times: list[float] = []
        self._start_time: float | None = None
        self._epoch_start_time: float | None = None
        self._total_samples = 0
        
    def start_epoch(self) -> None:
        """Call at the start of an epoch."""
        self._step_times.clear()
        self._total_samples = 0
        reset_gpu_peak_memory()
        self._epoch_start_time = time.perf_counter()
        
    @contextmanager
    def step(self, *, batch_size: int = 0) -> Iterable[None]:
        """Context manager for timing a single step."""
        _maybe_cuda_sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            _maybe_cuda_sync()
            self._step_times.append(time.perf_counter() - start)
            self._total_samples += batch_size
            
    def epoch_summary(self) -> dict[str, float]:
        """Return summary statistics for the epoch."""
        if not self._step_times:
            return {}
            
        sorted_times = sorted(self._step_times)
        n = len(sorted_times)
        total_time = sum(sorted_times)
        
        summary = {
            "num_steps": n,
            "total_time_s": total_time,
            "avg_step_ms": (total_time / n) * 1000,
            "p50_step_ms": _percentile(sorted_times, 50) * 1000,
            "p95_step_ms": _percentile(sorted_times, 95) * 1000,
            "min_step_ms": sorted_times[0] * 1000,
            "max_step_ms": sorted_times[-1] * 1000,
        }
        
        if self._total_samples > 0:
            summary["throughput_samples_per_s"] = self._total_samples / total_time
            
        # Add GPU memory stats
        mem_stats = get_gpu_memory_stats()
        if mem_stats:
            summary.update(mem_stats)
            
        return summary
        
    def log_epoch_summary(self, logger, *, epoch: int) -> None:
        """Log epoch summary to provided logger."""
        summary = self.epoch_summary()
        if not summary:
            return
            
        parts = [
            f"steps={summary.get('num_steps', 0)}",
            f"time={summary.get('total_time_s', 0):.1f}s",
            f"avg={summary.get('avg_step_ms', 0):.1f}ms",
            f"p50={summary.get('p50_step_ms', 0):.1f}ms",
            f"p95={summary.get('p95_step_ms', 0):.1f}ms",
        ]
        
        if "throughput_samples_per_s" in summary:
            parts.append(f"throughput={summary['throughput_samples_per_s']:.1f}img/s")
            
        if "peak_allocated_gib" in summary:
            parts.append(f"vram_peak={summary['peak_allocated_gib']:.2f}GiB")
            
        logger.info("[PERF][%s] epoch=%d %s", self.tag, epoch, " | ".join(parts))
