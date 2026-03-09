from __future__ import annotations

from typing import Mapping

import torch
import torch.distributed as dist


def is_dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not is_dist_ready():
        return 1
    return int(dist.get_world_size())


def get_rank() -> int:
    if not is_dist_ready():
        return 0
    return int(dist.get_rank())


def is_main_process() -> bool:
    return get_rank() == 0


def any_true(local_flag: bool, *, device: torch.device | None = None) -> bool:
    if not is_dist_ready():
        return bool(local_flag)
    dev = device or torch.device("cpu")
    val = torch.tensor(1 if local_flag else 0, device=dev, dtype=torch.int32)
    dist.all_reduce(val, op=dist.ReduceOp.MAX)
    return bool(int(val.item()) != 0)


def reduce_scalar(
    value: float,
    *,
    device: torch.device | None = None,
    average: bool = True,
) -> float:
    if not is_dist_ready():
        return float(value)
    dev = device or torch.device("cpu")
    t = torch.tensor(float(value), device=dev, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    if average:
        t = t / float(get_world_size())
    return float(t.item())


def reduce_metrics_dict(
    metrics: Mapping[str, float],
    *,
    device: torch.device | None = None,
    average: bool = True,
) -> dict[str, float]:
    if not is_dist_ready():
        return {k: float(v) for k, v in metrics.items()}
    reduced: dict[str, float] = {}
    for k in sorted(metrics.keys()):
        reduced[k] = reduce_scalar(
            float(metrics[k]),
            device=device,
            average=average,
        )
    return reduced
