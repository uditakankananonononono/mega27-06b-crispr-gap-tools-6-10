"""Guide-guide RNA duplex thermodynamics via ViennaRNA (RNAfold API).

ViennaRNA is the published standard (Lorenz 2011, Algorithms Mol Biol 6:26).
RNA.cofold computes the minimum-free-energy heteroduplex structure of two
strands; RNA.fold gives single-strand secondary-structure MFE (self-folding,
which competes with R-loop formation and Cas9 loading).
"""
from __future__ import annotations

import RNA


def _rna(seq: str) -> str:
    return seq.upper().replace("T", "U")


def heteroduplex_mfe(seq_a: str, seq_b: str) -> float:
    """MFE (kcal/mol) of the lowest-energy intermolecular duplex between two
    sequences, from ViennaRNA duplexfold (local alignment of both strands)."""
    d = RNA.duplexfold(_rna(seq_a), _rna(seq_b))
    return float(d.energy)


def cofold_mfe(seq_a: str, seq_b: str) -> float:
    """MFE of the joint structure of the two strands (includes intra terms)."""
    fc = RNA.fold_compound(_rna(seq_a) + "&" + _rna(seq_b))
    _, mfe = fc.mfe_dimer()
    return float(mfe)


def self_fold_mfe(seq: str) -> float:
    """Single-strand secondary-structure MFE (kcal/mol)."""
    _, mfe = RNA.fold(_rna(seq))
    return float(mfe)


def spacer_complementarity(seq_a: str, seq_b: str) -> float:
    """Fraction of positions in the best ungapped local complementarity run.

    A sliding-window exact-complement scan: the max over all offsets of the
    longest contiguous Watson-Crick-Franklin pairing run, normalized by spacer
    length. Captures seed-seed trans-binding that duplex MFE spreads out.
    """
    comp = {"A": "U", "U": "A", "G": "C", "C": "G"}
    a = _rna(seq_a)
    # antiparallel duplex: scan against the reverse complement of b
    b = "".join(comp[x] for x in _rna(seq_b))[::-1]
    best = 0
    for off in range(-len(b) + 1, len(a)):
        run = 0
        for i in range(len(a)):
            j = i - off
            if 0 <= j < len(b) and a[i] == b[j]:
                run += 1
                best = max(best, run)
            else:
                run = 0
    return best / max(len(a), 1)
