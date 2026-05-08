import numpy as np
import pytest

from driftvane import LatencyDrift
from driftvane.detectors.latency import ks_2samp


def test_identical_samples_have_zero_ks():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    sig = LatencyDrift().compute(a, a)
    assert sig.value == 0.0
    assert sig.metadata["ks_p_value"] == pytest.approx(1.0, abs=0.05)


def test_shifted_distribution_has_high_ks():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, size=500)
    b = rng.normal(loc=2.0, size=500)
    sig = LatencyDrift().compute(a, b)
    assert sig.value > 0.5
    assert sig.metadata["ks_p_value"] < 0.001


def test_threshold_flags_on_d_statistic():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, size=200)
    b = rng.normal(loc=2.0, size=200)
    sig = LatencyDrift(threshold=0.3).compute(a, b)
    assert sig.drifted is True


def test_p_threshold_flags():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, size=200)
    b = rng.normal(loc=2.0, size=200)
    sig = LatencyDrift(p_threshold=0.01).compute(a, b)
    assert sig.drifted is True


def test_cant_set_both_thresholds():
    with pytest.raises(ValueError, match="not both"):
        LatencyDrift(threshold=0.1, p_threshold=0.01)


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        ks_2samp([], [1.0, 2.0])


def test_ks_returns_in_unit_interval():
    rng = np.random.default_rng(0)
    a = rng.normal(size=100)
    b = rng.normal(size=100)
    d, p = ks_2samp(a, b)
    assert 0.0 <= d <= 1.0
    assert 0.0 <= p <= 1.0


def test_median_metadata_present():
    sig = LatencyDrift().compute([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
    assert sig.metadata["median_ref"] == pytest.approx(2.5)
    assert sig.metadata["median_cur"] == pytest.approx(25.0)
