"""Cross-species transfer evaluation API."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge

from crisprlib.benchmark import regression_metrics
from crisprlib.featurize import kmer_counts
from .divergence import species_shift


@dataclass
class TransferVerdict:
    species_a: str
    species_b: str
    heldout_spearman: float
    transfer_spearman: float
    transfer_drop: float
    shift: dict
    verdict: str


def quick_transfer_eval(contexts_a, y_a, contexts_b, y_b,
                        species_a="species_a", species_b="species_b",
                        seed: int = 0) -> TransferVerdict:
    """Ridge-kmer transfer evaluation (fast reference; CNN/GNN in benchmark script).

    Trains on 80% of A, tests on 20% of A and all of B.
    """
    rng = np.random.default_rng(seed)
    n = len(contexts_a)
    idx = rng.permutation(n)
    n_tr = int(0.8 * n)
    tr, te = idx[:n_tr], idx[n_tr:]
    X_a = np.stack([kmer_counts(s, 3) for s in contexts_a])
    X_b = np.stack([kmer_counts(s, 3) for s in contexts_b])
    y_a = np.asarray(y_a, dtype=np.float32)
    m = Ridge(alpha=1.0).fit(X_a[tr], y_a[tr])
    ho = regression_metrics(y_a[te], m.predict(X_a[te]))["spearman"]
    tf = regression_metrics(np.asarray(y_b, dtype=np.float32), m.predict(X_b))["spearman"]
    shift = species_shift(list(contexts_a), list(contexts_b))
    drop = ho - tf
    if drop > 0.15:
        verdict = (f"TRANSFER PENALTY: model loses {drop:.2f} Spearman from "
                   f"{species_a} to {species_b}; species-specific retraining advised.")
    elif drop >= -0.05:
        verdict = (f"TRANSFER OK: within +/-0.05 Spearman between {species_a} and "
                   f"{species_b} (drop {drop:.2f}).")
    else:
        verdict = (f"NO TRANSFER PENALTY: {species_b} score exceeds {species_a} "
                   f"held-out by {-drop:.2f} Spearman - label-noise difference likely, "
                   f"sequence rules transfer.")
    return TransferVerdict(species_a, species_b, round(ho, 4), round(tf, 4),
                           round(drop, 4), shift, verdict)
