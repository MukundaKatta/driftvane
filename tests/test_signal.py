from driftvane import DriftSignal


def test_to_dict_round_trips_fields():
    sig = DriftSignal(
        name="embedding_mmd",
        value=0.42,
        threshold=0.1,
        drifted=True,
        metadata={"n_ref": 100, "sigma": 1.5},
    )
    assert sig.to_dict() == {
        "name": "embedding_mmd",
        "value": 0.42,
        "threshold": 0.1,
        "drifted": True,
        "metadata": {"n_ref": 100, "sigma": 1.5},
    }


def test_defaults():
    sig = DriftSignal(name="x", value=0.0)
    assert sig.threshold is None
    assert sig.drifted is False
    assert sig.metadata == {}
