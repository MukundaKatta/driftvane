"""RetrievalDrift — measure shift in retriever output for the same queries.

Inputs are paired top-k document-id lists: for each query, the reference
retriever produced one ranked list and the current retriever produced another.
Drift = how much the top-k sets and rank order have moved.

Two metrics:
  * mean_jaccard_at_k: average Jaccard overlap of the top-k sets (1.0 = identical)
  * mean_rbo: rank-biased overlap, weights early positions more (1.0 = identical)

The reported drift value is 1 - mean_jaccard_at_k so that "more drift = larger
value" matches the convention in the other detectors.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from driftvane.detector import DriftSignal


def _jaccard(a: set[Any], b: set[Any]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _rbo(ref: Sequence[Any], cur: Sequence[Any], p: float = 0.9) -> float:
    """Rank-biased overlap. Weighted overlap of the two prefix sets at each depth.

    p controls how top-heavy the weighting is; p=0.9 puts ~86% of weight on the
    top 10. See Webber, Moffat, Zobel 2010.
    """
    depth = max(len(ref), len(cur))
    if depth == 0:
        return 1.0
    seen_ref: set[Any] = set()
    seen_cur: set[Any] = set()
    weighted_sum = 0.0
    weight_total = 0.0
    for i in range(depth):
        if i < len(ref):
            seen_ref.add(ref[i])
        if i < len(cur):
            seen_cur.add(cur[i])
        agreement = len(seen_ref & seen_cur) / (i + 1)
        w = p**i
        weighted_sum += agreement * w
        weight_total += w
    return weighted_sum / weight_total if weight_total > 0 else 1.0


class RetrievalDrift:
    """Detect retrieval drift across paired query→top-k results.

        rd = RetrievalDrift(k=10, threshold=0.3)
        signal = rd.compute(
            reference=[["doc_1", "doc_2", ...], ...],
            current=[["doc_1", "doc_3", ...], ...],
        )
    """

    def __init__(
        self,
        k: int = 10,
        threshold: float | None = None,
        name: str | None = None,
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.threshold = threshold
        self.name = name or f"retrieval_jaccard_at_{k}"

    def compute(
        self,
        reference: Sequence[Sequence[Any]],
        current: Sequence[Sequence[Any]],
    ) -> DriftSignal:
        if len(reference) != len(current):
            raise ValueError(
                f"reference and current must have the same number of queries; "
                f"got {len(reference)} vs {len(current)}"
            )
        if not reference:
            raise ValueError("need at least 1 query")

        jaccards: list[float] = []
        rbos: list[float] = []
        for ref_list, cur_list in zip(reference, current, strict=True):
            ref_top = list(ref_list[: self.k])
            cur_top = list(cur_list[: self.k])
            jaccards.append(_jaccard(set(ref_top), set(cur_top)))
            rbos.append(_rbo(ref_top, cur_top))

        mean_jaccard = sum(jaccards) / len(jaccards)
        mean_rbo = sum(rbos) / len(rbos)
        drift_value = 1.0 - mean_jaccard
        drifted = self.threshold is not None and drift_value > self.threshold

        return DriftSignal(
            name=self.name,
            value=drift_value,
            threshold=self.threshold,
            drifted=drifted,
            metadata={
                "n_queries": len(reference),
                "k": self.k,
                "mean_jaccard_at_k": mean_jaccard,
                "mean_rbo": mean_rbo,
            },
        )
