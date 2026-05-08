"""Core types: DriftSignal, DriftAlert."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DriftSignal:
    """One detector's verdict.

    name: stable identifier, e.g. "embedding_mmd", "retrieval_jaccard_at_10"
    value: the raw statistic
    threshold: the configured threshold; None means "report only, don't flag"
    drifted: True when value exceeds threshold
    metadata: detector-specific extras (sample sizes, kernel sigma, etc.)
    """

    name: str
    value: float
    threshold: float | None = None
    drifted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "drifted": self.drifted,
            "metadata": self.metadata,
        }


class DriftAlert(Exception):
    """Raised by DriftReport.alert_if when a threshold is breached."""

    def __init__(self, breaches: list[DriftSignal]):
        self.breaches = breaches
        names = ", ".join(f"{s.name}={s.value:.4f}>{s.threshold}" for s in breaches)
        super().__init__(f"drift detected: {names}")
