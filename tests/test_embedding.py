import numpy as np
import pytest

from driftvane import EmbeddingDrift
from driftvane.detectors.embedding import mmd_rbf


def _gen(n: int, dim: int, mean: float = 0.0, scale: float = 1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=scale, size=(n, dim))


def test_no_drift_for_same_distribution():
    a = _gen(200, 16, seed=1)
    b = _gen(200, 16, seed=2)
    sig = EmbeddingDrift().compute(a, b)
    # MMD between two N(0,1) samples of the same shape should be small
    assert sig.value < 0.05
    assert sig.metadata["n_ref"] == 200
    assert sig.metadata["dim"] == 16


def test_drift_for_shifted_mean():
    a = _gen(200, 16, mean=0.0, seed=1)
    b = _gen(200, 16, mean=2.0, seed=2)
    sig = EmbeddingDrift().compute(a, b)
    assert sig.value > 0.1


def test_threshold_flags_drift():
    a = _gen(200, 16, mean=0.0, seed=1)
    b = _gen(200, 16, mean=2.0, seed=2)
    sig = EmbeddingDrift(threshold=0.1).compute(a, b)
    assert sig.drifted is True


def test_threshold_passes_when_under():
    a = _gen(200, 16, seed=1)
    b = _gen(200, 16, seed=2)
    sig = EmbeddingDrift(threshold=0.5).compute(a, b)
    assert sig.drifted is False


def test_dim_mismatch_raises():
    a = _gen(50, 8)
    b = _gen(50, 16)
    with pytest.raises(ValueError, match="dim mismatch"):
        EmbeddingDrift().compute(a, b)


def test_one_sample_raises():
    a = _gen(1, 8)
    b = _gen(50, 8)
    with pytest.raises(ValueError, match="at least 2 samples"):
        EmbeddingDrift().compute(a, b)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        EmbeddingDrift(method="kld")


def test_explicit_sigma_used():
    a = _gen(50, 8)
    b = _gen(50, 8)
    sig = EmbeddingDrift(sigma=2.5).compute(a, b)
    assert sig.metadata["sigma"] == 2.5


def test_value_is_non_negative():
    a = _gen(200, 16, seed=1)
    b = _gen(200, 16, seed=1)
    sig = EmbeddingDrift().compute(a, b)
    assert sig.value >= 0.0


def test_mmd_rbf_returns_sigma_used():
    a = _gen(50, 8, seed=1)
    b = _gen(50, 8, seed=2)
    mmd2, sigma = mmd_rbf(a, b)
    assert mmd2 >= 0.0
    # median heuristic should pick a positive bandwidth
    assert sigma > 0.0


def test_mmd_rbf_respects_explicit_sigma():
    a = _gen(50, 8, seed=1)
    b = _gen(50, 8, seed=2)
    _, sigma = mmd_rbf(a, b, sigma=3.0)
    assert sigma == 3.0
