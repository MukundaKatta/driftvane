"""LatencyDrift — Kolmogorov-Smirnov two-sample test on latency arrays.

KS compares the empirical CDFs of two samples. The statistic is the maximum
absolute difference between the CDFs and is bounded in [0, 1]. It is robust
to scale and doesn't assume any particular distribution, which matches how
real LLM latency tails behave.

We compute the KS statistic from sorted arrays without scipy so the install
stays light. For an approximate p-value we use the standard asymptotic form
sqrt(-0.5 * ln(alpha/2) * (n1+n2)/(n1*n2)).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from driftvane.detector import DriftSignal


def ks_2samp(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Return (D, approx_p_value). Numpy-only two-sample KS."""
    a = np.sort(np.asarray(x, dtype=np.float64))
    b = np.sort(np.asarray(y, dtype=np.float64))
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        raise ValueError("both arrays must be non-empty")
    all_v = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_v, side="right") / n1
    cdf_b = np.searchsorted(b, all_v, side="right") / n2
    d = float(np.max(np.abs(cdf_a - cdf_b)))

    if d == 0.0:
        # asymptotic series degenerates at d=0; the null is trivially consistent
        return 0.0, 1.0

    en = math.sqrt(n1 * n2 / (n1 + n2))
    # asymptotic two-sided p-value (Smirnov)
    lam = (en + 0.12 + 0.11 / en) * d
    p = 2.0 * sum(((-1) ** (k - 1)) * math.exp(-2.0 * lam * lam * k * k) for k in range(1, 101))
    p = max(0.0, min(1.0, p))
    return d, p


class LatencyDrift:
    """Detect distribution shift in latency (or any 1-D numeric array).

        ld = LatencyDrift(threshold=0.2)        # threshold on KS statistic
        signal = ld.compute(reference=ref_lat, current=cur_lat)

    Or threshold on p-value:

        ld = LatencyDrift(p_threshold=0.01)
    """

    def __init__(
        self,
        threshold: float | None = None,
        p_threshold: float | None = None,
        name: str = "latency_ks",
    ) -> None:
        if threshold is not None and p_threshold is not None:
            raise ValueError("set either threshold or p_threshold, not both")
        self.threshold = threshold
        self.p_threshold = p_threshold
        self.name = name

    def compute(self, reference: Sequence[float], current: Sequence[float]) -> DriftSignal:
        d, p = ks_2samp(reference, current)
        if self.p_threshold is not None:
            drifted = p < self.p_threshold
        else:
            drifted = self.threshold is not None and d > self.threshold

        return DriftSignal(
            name=self.name,
            value=d,
            threshold=self.threshold,
            drifted=drifted,
            metadata={
                "n_ref": len(reference),
                "n_cur": len(current),
                "ks_p_value": p,
                "p_threshold": self.p_threshold,
                "median_ref": float(np.median(reference)),
                "median_cur": float(np.median(current)),
            },
        )
