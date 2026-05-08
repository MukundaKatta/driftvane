"""DriftReport — collect signals from multiple detectors."""

from __future__ import annotations

from typing import Any

from driftvane.detector import DriftAlert, DriftSignal


class DriftReport:
    """A bag of DriftSignals with output helpers.

    Build it incrementally:

        report = DriftReport()
        report.add(EmbeddingDrift().compute(ref_emb, cur_emb))
        report.add(LatencyDrift().compute(ref_lat, cur_lat))

    Or in one shot:

        report = DriftReport.from_signals([
            EmbeddingDrift().compute(ref_emb, cur_emb),
            LatencyDrift().compute(ref_lat, cur_lat),
        ])
    """

    def __init__(self) -> None:
        self._signals: list[DriftSignal] = []

    @classmethod
    def from_signals(cls, signals: list[DriftSignal]) -> DriftReport:
        r = cls()
        for s in signals:
            r.add(s)
        return r

    def add(self, signal: DriftSignal) -> DriftReport:
        self._signals.append(signal)
        return self

    @property
    def signals(self) -> list[DriftSignal]:
        return list(self._signals)

    def get(self, name: str) -> DriftSignal | None:
        for s in self._signals:
            if s.name == name:
                return s
        return None

    def any_drifted(self) -> bool:
        return any(s.drifted for s in self._signals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signals": [s.to_dict() for s in self._signals],
            "any_drifted": self.any_drifted(),
        }

    def to_pandas(self):
        # imported lazily so pandas isn't required for non-DataFrame users
        import pandas as pd

        if not self._signals:
            return pd.DataFrame(columns=["name", "value", "threshold", "drifted"])
        return pd.DataFrame(
            [
                {
                    "name": s.name,
                    "value": s.value,
                    "threshold": s.threshold,
                    "drifted": s.drifted,
                    **{f"meta_{k}": v for k, v in s.metadata.items()},
                }
                for s in self._signals
            ]
        )

    def alert_if(self, thresholds: dict[str, float]) -> None:
        """Raise DriftAlert if any of the given signals exceeds its threshold.

        Overrides the threshold each signal was computed with. Use this when the
        report is being evaluated against a different policy than the detector
        was constructed with (e.g. CI vs. prod).
        """
        breaches = []
        for s in self._signals:
            if s.name in thresholds and s.value > thresholds[s.name]:
                breaches.append(
                    DriftSignal(
                        name=s.name,
                        value=s.value,
                        threshold=thresholds[s.name],
                        drifted=True,
                        metadata=s.metadata,
                    )
                )
        if breaches:
            raise DriftAlert(breaches)
