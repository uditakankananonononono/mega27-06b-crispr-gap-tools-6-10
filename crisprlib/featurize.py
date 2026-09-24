"""Sequence featurization for CRISPR guide/off-target modelling.

All encodings are deterministic and dependency-light (numpy only at this layer).
Conventions:
- DNA alphabet order A, C, G, T (index 0..3). Unknown bases -> all-zero row.
- Sequences are padded/trimmed from the 3' (PAM-distal) side unless stated.
"""
from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np

DNA_ALPHABET = "ACGT"
_BASE_TO_IDX = {b: i for i, b in enumerate(DNA_ALPHABET)}

# RNA:DNA / DNA:DNA nearest-neighbour thermodynamic parameters (Sugimoto-class,
# kcal/mol at 37C), used for guide:target duplex stability features. Values are
# the canonical DNA duplex NN parameters from SantaLucia 1998 (unified set).
NN_DG = {
    "AA": -1.00, "AC": -1.44, "AG": -1.28, "AT": -0.88,
    "CA": -1.45, "CC": -1.84, "CG": -2.17, "CT": -1.28,
    "GA": -1.30, "GC": -2.24, "GG": -1.84, "GT": -1.44,
    "TA": -0.58, "TC": -1.30, "TG": -1.45, "TT": -1.00,
}


def one_hot(seq: str, length: int | None = None) -> np.ndarray:
    """(L, 4) float32 one-hot encoding; N or unknown -> zero row."""
    seq = seq.upper()
    L = length if length is not None else len(seq)
    out = np.zeros((L, 4), dtype=np.float32)
    for i, b in enumerate(seq[:L]):
        j = _BASE_TO_IDX.get(b)
        if j is not None:
            out[i, j] = 1.0
    return out


def dinuc_one_hot(seq: str, length: int | None = None) -> np.ndarray:
    """(L-1, 16) dinucleotide one-hot encoding."""
    seq = seq.upper()
    L = (length if length is not None else len(seq)) - 1
    out = np.zeros((max(L, 0), 16), dtype=np.float32)
    for i in range(min(len(seq) - 1, max(L, 0))):
        a = _BASE_TO_IDX.get(seq[i])
        b = _BASE_TO_IDX.get(seq[i + 1])
        if a is not None and b is not None:
            out[i, a * 4 + b] = 1.0
    return out


def gc_content(seq: str) -> float:
    s = [b for b in seq.upper() if b in _BASE_TO_IDX]
    if not s:
        return 0.0
    return sum(1 for b in s if b in "GC") / len(s)


def thermo_dg(seq: str) -> float:
    """Duplex free energy (kcal/mol) via SantaLucia unified NN parameters.

    Unknown dinucleotides contribute 0. Returns 0.0 for sequences < 2 nt.
    """
    seq = seq.upper()
    return float(sum(NN_DG.get(seq[i:i + 2], 0.0) for i in range(len(seq) - 1)))


def position_features(seq: str) -> np.ndarray:
    """(L, 3) per-position [GC flag, purine flag, position-in-seq normalized]."""
    seq = seq.upper()
    L = len(seq)
    out = np.zeros((L, 3), dtype=np.float32)
    for i, b in enumerate(seq):
        out[i, 0] = 1.0 if b in "GC" else 0.0
        out[i, 1] = 1.0 if b in "AG" else 0.0
        out[i, 2] = i / max(L - 1, 1)
    return out


def kmer_counts(seq: str, k: int = 3) -> np.ndarray:
    """4^k kmer count vector (canonical forward orientation)."""
    n = 4 ** k
    out = np.zeros(n, dtype=np.float32)
    seq = seq.upper()
    for i in range(len(seq) - k + 1):
        idx = 0
        ok = True
        for b in seq[i:i + k]:
            j = _BASE_TO_IDX.get(b)
            if j is None:
                ok = False
                break
            idx = idx * 4 + j
        if ok:
            out[idx] += 1.0
    return out


def align_pair(guide: str, target: str) -> tuple[np.ndarray, np.ndarray]:
    """Encode a guide/off-target pair: (L,4) guide one-hot and (L,) mismatch vector.

    Mismatch vector marks positions where guide and target differ (1.0 = mismatch,
    0.0 = match or unaligned padding).
    """
    L = max(len(guide), len(target))
    g = one_hot(guide, L)
    mm = np.zeros(L, dtype=np.float32)
    for i in range(min(len(guide), len(target))):
        if guide[i].upper() != target[i].upper():
            mm[i] = 1.0
    return g, mm


def microhomology_score(left: str, right: str, max_len: int = 20) -> float:
    """Length-weighted microhomology score between flanks of a cut site.

    Sums 1/k over all exact common substrings anchored at the break, the
    scoring style used by MMEJ predictors (Bae 2014-class).
    """
    score = 0.0
    for k in range(2, min(len(left), len(right), max_len) + 1):
        if left[-k:] == right[:k]:
            score += 1.0 / k
    return score


ALL_KMERS_3 = ["".join(t) for t in itertools.product(DNA_ALPHABET, repeat=3)]
