import pytest

from driftvane import RetrievalDrift


def test_identical_rankings_have_zero_drift():
    ref = [["a", "b", "c", "d"], ["x", "y", "z"]]
    cur = [["a", "b", "c", "d"], ["x", "y", "z"]]
    sig = RetrievalDrift(k=4).compute(ref, cur)
    assert sig.value == 0.0
    assert sig.metadata["mean_jaccard_at_k"] == 1.0
    assert sig.metadata["mean_rbo"] == 1.0


def test_disjoint_rankings_have_max_drift():
    ref = [["a", "b", "c"]]
    cur = [["x", "y", "z"]]
    sig = RetrievalDrift(k=3).compute(ref, cur)
    assert sig.value == 1.0
    assert sig.metadata["mean_jaccard_at_k"] == 0.0


def test_partial_overlap():
    ref = [["a", "b", "c", "d"]]
    cur = [["a", "b", "x", "y"]]
    sig = RetrievalDrift(k=4).compute(ref, cur)
    # Jaccard of {a,b,c,d} vs {a,b,x,y} = 2/6 = 1/3
    assert sig.metadata["mean_jaccard_at_k"] == pytest.approx(1 / 3)


def test_threshold_flags():
    ref = [["a", "b", "c"]]
    cur = [["a", "x", "y"]]
    sig = RetrievalDrift(k=3, threshold=0.5).compute(ref, cur)
    # 1 - (1/5) = 0.8 > 0.5
    assert sig.drifted is True


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same number of queries"):
        RetrievalDrift().compute([["a"]], [["a"], ["b"]])


def test_empty_input_raises():
    with pytest.raises(ValueError, match="at least 1 query"):
        RetrievalDrift().compute([], [])


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="k must be"):
        RetrievalDrift(k=0)


def test_rbo_weights_top_positions_more():
    # Same set, different order: Jaccard sees no drift, RBO does
    ref = [["a", "b", "c", "d", "e"]]
    cur = [["e", "d", "c", "b", "a"]]
    sig = RetrievalDrift(k=5).compute(ref, cur)
    assert sig.metadata["mean_jaccard_at_k"] == 1.0
    assert sig.metadata["mean_rbo"] < 1.0
