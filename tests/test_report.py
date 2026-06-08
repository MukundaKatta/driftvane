import pytest

from driftvane import DriftAlert, DriftReport, DriftSignal


def _sig(
    name: str,
    value: float,
    threshold: float | None = None,
    drifted: bool = False,
) -> DriftSignal:
    return DriftSignal(name=name, value=value, threshold=threshold, drifted=drifted)


def test_empty_report_is_not_drifted():
    r = DriftReport()
    assert r.signals == []
    assert r.any_drifted() is False
    assert r.to_dict() == {"signals": [], "any_drifted": False}


def test_add_chains():
    r = DriftReport()
    out = r.add(_sig("a", 0.1)).add(_sig("b", 0.2))
    assert out is r
    assert [s.name for s in r.signals] == ["a", "b"]


def test_from_signals():
    r = DriftReport.from_signals([_sig("a", 0.1), _sig("b", 0.2, threshold=0.1, drifted=True)])
    assert r.any_drifted() is True
    assert r.get("a").value == 0.1
    assert r.get("missing") is None


def test_to_pandas_includes_metadata_columns():
    r = DriftReport().add(DriftSignal(name="x", value=0.5, metadata={"sigma": 1.5, "n": 100}))
    df = r.to_pandas()
    assert list(df.columns) == ["name", "value", "threshold", "drifted", "meta_sigma", "meta_n"]
    assert df.iloc[0]["meta_sigma"] == 1.5


def test_to_pandas_empty_report_has_base_columns():
    df = DriftReport().to_pandas()
    assert list(df.columns) == ["name", "value", "threshold", "drifted"]
    assert len(df) == 0


def test_alert_if_raises_on_breach():
    r = DriftReport().add(_sig("emb", 0.3)).add(_sig("lat", 0.05))
    with pytest.raises(DriftAlert) as exc:
        r.alert_if({"emb": 0.2, "lat": 0.1})
    assert len(exc.value.breaches) == 1
    assert exc.value.breaches[0].name == "emb"


def test_alert_if_no_breach():
    r = DriftReport().add(_sig("emb", 0.05))
    # should not raise
    r.alert_if({"emb": 0.2})


def test_alert_if_ignores_unlisted_signals():
    r = DriftReport().add(_sig("emb", 999.0))
    r.alert_if({"latency": 0.1})  # emb has no threshold here, should not raise
