"""driftvane — compose drift detectors for RAG and agent systems.

A small library that lets you wire up multiple drift signals (embedding,
retrieval, response, latency) into one DriftReport. No server, no UI.
"""

from driftvane.detector import DriftAlert, DriftSignal
from driftvane.detectors.embedding import EmbeddingDrift
from driftvane.detectors.latency import LatencyDrift
from driftvane.detectors.response import ResponseDrift
from driftvane.detectors.retrieval import RetrievalDrift
from driftvane.report import DriftReport

__version__ = "0.1.0"

__all__ = [
    "DriftAlert",
    "DriftReport",
    "DriftSignal",
    "EmbeddingDrift",
    "LatencyDrift",
    "ResponseDrift",
    "RetrievalDrift",
    "__version__",
]
