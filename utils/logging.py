"""Training logging utilities."""
import logging
import os
import sys
import time
from typing import Any, Dict, Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class MetricsTracker:
    """Tracks running averages of training metrics."""

    def __init__(self):
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._t0 = time.time()

    def update(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            self._sums[k]   = self._sums.get(k, 0.0) + v
            self._counts[k] = self._counts.get(k, 0) + 1

    def averages(self) -> Dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()
        self._t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self._t0
