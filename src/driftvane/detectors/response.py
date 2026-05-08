"""ResponseDrift — answer-vs-context grounding drift across batches.

For each (intent, context, answer) triple, compute Jaccard overlap of token
sets between the answer and the context. Then compare the *distribution* of
those scores between reference and current batches.

The drift value is the absolute difference of the mean grounding scores. A
shrinking mean answer-to-context overlap is the signal you want to catch:
the model is wandering off the retrieved context.

If `context-drift-detector-py` is installed we delegate per-triple scoring
to it for compatibility with that library's signal definitions; otherwise
we use the inline tokenizer below. Either way, the aggregation is ours.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from driftvane.detector import DriftSignal


@dataclass(frozen=True)
class Triple:
    intent: str
    context: str | list[str]
    answer: str


_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _flatten_context(ctx: str | Iterable[str]) -> str:
    if isinstance(ctx, str):
        return ctx
    return " ".join(ctx)


def _grounding_score(triple: Triple) -> float:
    """answer ∩ context / answer (recall-style; 1.0 = fully grounded)."""
    ans = _tokens(triple.answer)
    if not ans:
        return 1.0
    ctx = _tokens(_flatten_context(triple.context))
    return len(ans & ctx) / len(ans)


def _try_load_external_scorer():
    try:
        from context_drift_detector import detect  # type: ignore
    except ImportError:
        return None

    def _score(triple: Triple) -> float:
        ctx = triple.context if isinstance(triple.context, list) else [triple.context]
        result = detect(triple.intent, ctx, triple.answer)
        # context-drift-detector-py exposes signals dict with answer_to_context
        return float(result.signals.get("answer_to_context", _grounding_score(triple)))

    return _score


class ResponseDrift:
    """Detect drift in how well answers stay grounded in retrieved context.

        rsp = ResponseDrift(threshold=0.15)
        signal = rsp.compute(
            reference=[Triple("...", "...", "..."), ...],
            current=[Triple("...", "...", "..."), ...],
        )

    Pass `use_external=False` to force the inline tokenizer even when
    context-drift-detector-py is installed.
    """

    def __init__(
        self,
        threshold: float | None = None,
        name: str = "response_grounding_shift",
        use_external: bool = True,
    ) -> None:
        self.threshold = threshold
        self.name = name
        self.use_external = use_external
        self._scorer = _try_load_external_scorer() if use_external else None

    def compute(
        self,
        reference: Iterable[Triple | dict],
        current: Iterable[Triple | dict],
    ) -> DriftSignal:
        ref = [t if isinstance(t, Triple) else Triple(**t) for t in reference]
        cur = [t if isinstance(t, Triple) else Triple(**t) for t in current]
        if not ref or not cur:
            raise ValueError("need at least 1 triple in each batch")

        score = self._scorer or _grounding_score
        ref_scores = [score(t) for t in ref]
        cur_scores = [score(t) for t in cur]

        mean_ref = sum(ref_scores) / len(ref_scores)
        mean_cur = sum(cur_scores) / len(cur_scores)
        # we care about *worsening* grounding, so use signed shift but report
        # absolute as the drift value
        signed_shift = mean_cur - mean_ref
        drift_value = abs(signed_shift)
        drifted = self.threshold is not None and drift_value > self.threshold

        return DriftSignal(
            name=self.name,
            value=drift_value,
            threshold=self.threshold,
            drifted=drifted,
            metadata={
                "n_ref": len(ref),
                "n_cur": len(cur),
                "mean_ref_grounding": mean_ref,
                "mean_cur_grounding": mean_cur,
                "signed_shift": signed_shift,
                "scorer": "external" if self._scorer else "inline_jaccard",
            },
        )
