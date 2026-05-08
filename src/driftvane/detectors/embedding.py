"""EmbeddingDrift — Maximum Mean Discrepancy with RBF kernel.

MMD is a kernel two-sample test. It tests whether two batches of embeddings
were drawn from the same distribution. MMD^2 is zero when the distributions
match and grows with the distance between them.

We compute the squared MMD with the RBF (Gaussian) kernel:
    k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    MMD^2 = E[k(X, X')] + E[k(Y, Y')] - 2 E[k(X, Y)]

When sigma is None we use the median heuristic on the merged sample, which
is the standard default and removes the main hyperparameter footgun.

Cost is O(n^2) memory and time, so call this with batches up to a few
thousand vectors. For larger sets, subsample first.
"""

from __future__ import annotations

import numpy as np

from driftvane.detector import DriftSignal


def _pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix, shape (len(a), len(b))."""
    a2 = np.sum(a * a, axis=1)[:, None]
    b2 = np.sum(b * b, axis=1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * a @ b.T, 0.0)


def _median_heuristic_sigma(x: np.ndarray, y: np.ndarray) -> float:
    """Median pairwise distance on the merged sample. Robust default for sigma."""
    z = np.concatenate([x, y], axis=0)
    # subsample to keep this cheap on big inputs
    if len(z) > 1000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(z), size=1000, replace=False)
        z = z[idx]
    d2 = _pairwise_sq_dists(z, z)
    iu = np.triu_indices_from(d2, k=1)
    median_sq = float(np.median(d2[iu]))
    # sigma is the bandwidth, not sigma^2; floor to avoid div-by-zero
    return max(np.sqrt(median_sq / 2.0), 1e-8)


def mmd_rbf(x: np.ndarray, y: np.ndarray, sigma: float | None = None) -> tuple[float, float]:
    """Compute MMD^2 between two batches with RBF kernel.

    Returns (mmd_squared, sigma_used).
    """
    if sigma is None:
        sigma = _median_heuristic_sigma(x, y)
    gamma = 1.0 / (2.0 * sigma * sigma)

    kxx = np.exp(-gamma * _pairwise_sq_dists(x, x))
    kyy = np.exp(-gamma * _pairwise_sq_dists(y, y))
    kxy = np.exp(-gamma * _pairwise_sq_dists(x, y))

    mmd2 = float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())
    # numerical noise can push the value slightly negative; clamp at 0
    return max(mmd2, 0.0), sigma


class EmbeddingDrift:
    """Detect distribution shift between two batches of embedding vectors.

        ed = EmbeddingDrift(threshold=0.1)
        signal = ed.compute(reference=ref_emb, current=cur_emb)
    """

    def __init__(
        self,
        method: str = "mmd",
        sigma: float | None = None,
        threshold: float | None = None,
        name: str = "embedding_mmd",
    ) -> None:
        if method != "mmd":
            raise ValueError(f"unknown method: {method!r}; only 'mmd' is supported")
        self.method = method
        self.sigma = sigma
        self.threshold = threshold
        self.name = name

    def compute(self, reference: np.ndarray, current: np.ndarray) -> DriftSignal:
        ref = np.asarray(reference, dtype=np.float64)
        cur = np.asarray(current, dtype=np.float64)
        if ref.ndim != 2 or cur.ndim != 2:
            raise ValueError("reference and current must be 2-D (n_samples, n_dims)")
        if ref.shape[1] != cur.shape[1]:
            raise ValueError(
                f"dim mismatch: reference has {ref.shape[1]}, current has {cur.shape[1]}"
            )
        if len(ref) < 2 or len(cur) < 2:
            raise ValueError("need at least 2 samples in each set")

        value, sigma_used = mmd_rbf(ref, cur, sigma=self.sigma)
        drifted = self.threshold is not None and value > self.threshold
        return DriftSignal(
            name=self.name,
            value=value,
            threshold=self.threshold,
            drifted=drifted,
            metadata={
                "n_ref": int(ref.shape[0]),
                "n_cur": int(cur.shape[0]),
                "dim": int(ref.shape[1]),
                "sigma": float(sigma_used),
                "method": self.method,
            },
        )
