# driftvane

[![CI](https://github.com/MukundaKatta/driftvane/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/driftvane/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/driftvane.svg)](https://pypi.org/project/driftvane/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Compose drift detectors for RAG and agent systems.**

Most drift libraries are either tabular-only (Evidently, DataDrift) or are
platforms that want you to ship telemetry to their backend (Phoenix, Arize).
`driftvane` is a small Python library that lets you wire up multiple drift
signals — embedding, retrieval, response, latency — into one report. No
server, no UI, no telemetry. Plug it into a Lambda or Glue job, get a
`pandas.DataFrame` or a JSON dict back.

## Install

```bash
pip install driftvane
# optional
pip install "driftvane[pandas]"            # to_pandas()
pip install "driftvane[external-response]" # delegate response scoring to context-drift-detector-py
```

## Quickstart

```python
import numpy as np
from driftvane import (
    DriftReport,
    EmbeddingDrift,
    RetrievalDrift,
    ResponseDrift,
    LatencyDrift,
)
from driftvane.detectors.response import Triple

ref_emb = np.load("reference_query_embeddings.npy")  # (n, 768)
cur_emb = np.load("current_query_embeddings.npy")

report = DriftReport.from_signals([
    EmbeddingDrift(threshold=0.1).compute(ref_emb, cur_emb),
    RetrievalDrift(k=10, threshold=0.3).compute(ref_top_k, cur_top_k),
    ResponseDrift(threshold=0.15).compute(ref_triples, cur_triples),
    LatencyDrift(p_threshold=0.01).compute(ref_latencies, cur_latencies),
])

if report.any_drifted():
    print(report.to_pandas())
```

Or fail a CI job when retrieval moves too much:

```python
from driftvane import DriftAlert

try:
    report.alert_if({"retrieval_jaccard_at_10": 0.2})
except DriftAlert as e:
    sys.exit(f"drift gate failed: {e}")
```

## Detectors

| Detector | Input | Statistic | Notes |
|---|---|---|---|
| `EmbeddingDrift` | two `(n, d)` arrays | MMD with RBF kernel, median-heuristic sigma | numpy-only, O(n²) — subsample for n > a few thousand |
| `RetrievalDrift` | paired top-k id lists | 1 − mean Jaccard@k; reports RBO too | aligned queries required |
| `ResponseDrift` | `(intent, context, answer)` triples | shift in mean answer-to-context grounding | uses `context-drift-detector-py` if installed |
| `LatencyDrift` | two 1-D arrays of floats | Kolmogorov–Smirnov D + asymptotic p-value | scipy-free |

Each detector returns a `DriftSignal(name, value, threshold, drifted, metadata)`.
`DriftReport` collects them.

## What it does NOT do

- No server. No UI. No telemetry shipping.
- No tabular feature drift — use [DataDrift](https://github.com/MukundaKatta/DataDrift)
  for KS/PSI on classical features.
- No live trace ingestion or OTel collection — point this at parquet/numpy
  arrays you already have.
- No causal root-cause analysis. It tells you *that* drift is there, not why.
- No model retraining triggers — emit your own when `report.any_drifted()`.

## Why not Phoenix / Arize / Evidently / Ragas?

| | driftvane | Phoenix | Arize | Evidently | Ragas |
|---|---|---|---|---|---|
| Library-only (no server) | ✓ | ✗ | ✗ | partial | ✓ |
| RAG-shaped detectors | ✓ | ✓ | ✓ | ✗ | ✓ |
| Embedding MMD out of the box | ✓ | partial | ✓ | ✗ | ✗ |
| Retrieval rank-shift | ✓ | ✗ | partial | ✗ | ✗ |
| Run inside a 5s Lambda | ✓ | ✗ | ✗ | ✓ | partial |
| numpy-only core deps | ✓ | ✗ | ✗ | ✗ | ✗ |

## Status

v0.1 — alpha. The four detectors above work and have tests. Public API may
change before v1.0. Issues and PRs welcome.
