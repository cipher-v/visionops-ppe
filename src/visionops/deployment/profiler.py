"""Small non-overlapping stage latency accumulator."""

from __future__ import annotations

import statistics
from collections import defaultdict


class StageProfiler:
    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = defaultdict(list)

    def record(self, stage: str, elapsed_seconds: float) -> None:
        if not stage or elapsed_seconds < 0:
            raise ValueError("Profiler stage and duration must be valid")
        self._samples[stage].append(elapsed_seconds)

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float:
        ordered = sorted(values)
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    def summary(self) -> dict[str, dict[str, float | int]]:
        return {
            stage: {
                "count": len(values),
                "total_ms": sum(values) * 1000,
                "mean_ms": statistics.fmean(values) * 1000,
                "p50_ms": statistics.median(values) * 1000,
                "p95_ms": self._percentile(values, 0.95) * 1000,
            }
            for stage, values in sorted(self._samples.items())
            if values
        }
