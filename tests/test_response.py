import pytest

from driftvane import ResponseDrift
from driftvane.detectors.response import Triple


def test_grounded_answers_have_no_drift():
    triples = [
        Triple(
            intent="What is the capital of France?",
            context="Paris is the capital of France.",
            answer="Paris is the capital of France.",
        ),
        Triple(
            intent="What is 2+2?",
            context="Two plus two equals four.",
            answer="Two plus two equals four.",
        ),
    ]
    sig = ResponseDrift(use_external=False).compute(triples, triples)
    assert sig.value == pytest.approx(0.0)


def test_ungrounded_current_drifts_negative():
    grounded = [
        Triple(
            intent="capital of France",
            context="Paris is the capital of France.",
            answer="Paris is the capital of France.",
        )
    ]
    ungrounded = [
        Triple(
            intent="capital of France",
            context="Paris is the capital of France.",
            answer="Wombats live in Australia.",
        )
    ]
    sig = ResponseDrift(use_external=False).compute(grounded, ungrounded)
    assert sig.metadata["signed_shift"] < 0
    assert sig.value > 0


def test_threshold_flags():
    a = [Triple("q", "the answer is forty two", "the answer is forty two")]
    b = [Triple("q", "the answer is forty two", "completely unrelated text")]
    sig = ResponseDrift(threshold=0.3, use_external=False).compute(a, b)
    assert sig.drifted is True


def test_accepts_dicts_and_list_context():
    a = [{"intent": "q", "context": ["fact a", "fact b"], "answer": "fact a"}]
    b = [{"intent": "q", "context": ["fact a"], "answer": "fact b fact a"}]
    sig = ResponseDrift(use_external=False).compute(a, b)
    assert sig.metadata["n_ref"] == 1
    assert sig.metadata["n_cur"] == 1


def test_empty_raises():
    with pytest.raises(ValueError, match="at least 1 triple"):
        ResponseDrift(use_external=False).compute([], [])


def test_inline_scorer_used_when_external_disabled():
    triples = [Triple("q", "ctx", "ctx")]
    sig = ResponseDrift(use_external=False).compute(triples, triples)
    assert sig.metadata["scorer"] == "inline_jaccard"
