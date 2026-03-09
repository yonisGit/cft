from __future__ import annotations

import logging
import os
import sys
import math
from typing import Optional, Union


class _ExactLevelFilter(logging.Filter):
    def __init__(self, level: int) -> None:
        super().__init__()
        self._level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == self._level


class _NotLevelFilter(logging.Filter):
    def __init__(self, level: int) -> None:
        super().__init__()
        self._level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno != self._level


class _AllowlistNameFilter(logging.Filter):
    def __init__(self, prefixes: tuple[str, ...]) -> None:
        super().__init__()
        self._prefixes = prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(self._prefixes)


def _parse_level(level: Union[str, int, None]) -> int:
    if level is None:
        level = os.getenv("CFT_LOG_LEVEL", "DEBUG")
    if isinstance(level, int):
        return level
    text = str(level).strip()
    if text.isdigit():
        return int(text)
    return logging._nameToLevel.get(text.upper(), logging.INFO)


def setup_logging(
    *,
    level: Union[str, int, None] = None,
    force: bool = False,
) -> None:
    root = logging.getLogger()
    if root.handlers and not force:
        return

    for handler in list(root.handlers):
        root.removeHandler(handler)

    root.setLevel(_parse_level(level))

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.addFilter(_ExactLevelFilter(logging.INFO))
    stdout_handler.addFilter(_AllowlistNameFilter(("cft", "__main__")))
    stdout_handler.setFormatter(logging.Formatter("%(message)s"))

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.DEBUG)
    stderr_handler.addFilter(_NotLevelFilter(logging.INFO))
    stderr_handler.addFilter(_AllowlistNameFilter(("cft", "__main__")))
    stderr_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )

    root.addHandler(stdout_handler)
    root.addHandler(stderr_handler)


def resolve_progress_log_every(*, total_steps: int, default_every: int) -> int:
    total_steps = int(total_steps)
    if total_steps <= 0:
        return 1

    every_override = os.getenv("CFT_PROGRESS_LOG_EVERY")
    if every_override is not None:
        every = int(every_override)
        if every < 1:
            raise ValueError(f"CFT_PROGRESS_LOG_EVERY must be >= 1, got {every}.")
        return every

    frac_override = os.getenv("CFT_PROGRESS_LOG_FRAC")
    if frac_override is not None:
        frac = float(frac_override)
        if frac <= 0:
            raise ValueError(f"CFT_PROGRESS_LOG_FRAC must be > 0, got {frac}.")
        return max(1, int(math.ceil(total_steps * frac)))

    return max(1, int(default_every))


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if not name:
        return logging.getLogger("cft")
    if name.startswith(("cft", "__main__")):
        return logging.getLogger(name)
    return logging.getLogger(f"cft.{name}")
