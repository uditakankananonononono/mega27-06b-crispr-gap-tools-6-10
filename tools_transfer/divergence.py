"""Feature-space divergence between species' guide sets (transfer diagnostics)."""
from __future__ import annotations

import numpy as np

from crisprlib.featurize import gc_content, kmer_counts


def _kmer_dist(seqs, k=3) -> np.ndarray:
    v = np.mean([kmer_counts(s, k) for s in seqs], axis=0)
    return v / max(v.sum(), 1e-9)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (nats) between two distributions."""
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def species_shift(seqs_a: list[str], seqs_b: list[str], k: int = 3) -> dict:
    gc_a = float(np.mean([gc_content(s) for s in seqs_a]))
    gc_b = float(np.mean([gc_content(s) for s in seqs_b]))
    js = js_divergence(_kmer_dist(seqs_a, k), _kmer_dist(seqs_b, k))
    return {"gc_mean_a": gc_a, "gc_mean_b": gc_b, "gc_delta": abs(gc_a - gc_b),
            "kmer_js_divergence": js, "k": k}
