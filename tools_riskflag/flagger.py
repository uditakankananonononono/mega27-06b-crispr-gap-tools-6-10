"""Mechanistic risk flagging for large on-target deletions and rearrangements.

Grounded in published findings:
- Kosicki et al. 2018 (Nat Biotechnol 36:765, nbt.4192): Cas9 breaks in mESCs
  produce kilobase-scale deletions and complex rearrangements in a substantial
  fraction of edited cells; lesions extend asymmetrically around the break.
- Wen et al. 2022 (Sci Adv 8:eabo7676): large gene modifications (deletions,
  insertions, inversions) occur at appreciable rates at on-target sites and are
  missed by standard short-range PCR genotyping.

Risk features (all computed from the local sequence, no black box):
1. Microhomology density around the cut: MMEJ-mediated end joining between
   direct repeats flanking the break is the mechanistic driver of long
   deletions. We count direct-repeat pairs (>=8 bp exact) with one copy on each
   side of the cut within the scan window.
2. Repeat span: the largest repeat-pair span - the worst-case deletion length.
3. Low-complexity sequence (Shannon entropy of 10-mers, homopolymer runs):
   repetitive DNA is rearrangement-prone.
4. Asymmetry: left-vs-right repeat density skew.

Score: weighted sum mapped to LOW / MODERATE / HIGH flags. This is a
mechanistic screen, not a trained probability: no large per-sequence
quantitative dataset exists (documented in the paper), and we do not claim
calibrated rates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

WINDOW = 250          # bp scanned each side of the cut
MIN_REPEAT = 8        # minimum exact direct repeat length
MAX_REPEAT = 30


@dataclass
class RiskReport:
    n_repeat_pairs: int
    max_deletion_span: int
    left_repeat_density: float
    right_repeat_density: float
    asymmetry: float
    mean_entropy: float
    max_homopolymer: int
    score: float
    flag: str
    details: dict = field(default_factory=dict)


def _direct_repeats(seq: str, min_len: int = MIN_REPEAT, max_len: int = MAX_REPEAT):
    """All (i, j, L) with seq[i:i+L] == seq[j:j+L], i < j, L in [min_len, max_len]."""
    hits = []
    n = len(seq)
    for L in range(min_len, max_len + 1):
        seen: dict[str, int] = {}
        for j in range(n - L + 1):
            k = seq[j:j + L]
            if "N" in k:
                continue
            if k in seen:
                hits.append((seen[k], j, L))
            else:
                seen[k] = j
    return hits


def _shannon_entropy(seq: str, k: int = 10) -> float:
    if len(seq) < k:
        return 0.0
    from collections import Counter
    kmers = Counter(seq[i:i + k] for i in range(len(seq) - k + 1))
    total = sum(kmers.values())
    return -sum((c / total) * math.log2(c / total) for c in kmers.values())


def _max_homopolymer(seq: str) -> int:
    best = run = 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def flag_cutsite(context: str, cut_pos: int, window: int = WINDOW) -> RiskReport:
    """Flag a cut site inside a local genomic context.

    context: genomic sequence containing the cut site (uppercase ACGTN).
    cut_pos: index of the cut (between cut_pos-1 and cut_pos).
    """
    ctx = context.upper()
    lo = max(0, cut_pos - window)
    hi = min(len(ctx), cut_pos + window)
    left = ctx[lo:cut_pos]
    right = ctx[cut_pos:hi]
    region = ctx[lo:hi]

    reps = _direct_repeats(region)
    # keep pairs spanning the cut (repeat copy on each side)
    cut_local = cut_pos - lo
    spanning = [(i, j, L) for i, j, L in reps
                if i < cut_local and j >= cut_local]
    spans = [(j + L) - i for i, j, L in spanning]
    max_span = max(spans) if spans else 0

    reps_left = [(i, j, L) for i, j, L in reps if j < cut_local]
    reps_right = [(i, j, L) for i, j, L in reps if i >= cut_local]
    ld = len(reps_left) / max(len(left), 1)
    rd = len(reps_right) / max(len(right), 1)
    asym = abs(ld - rd) / max(ld + rd, 1e-9)

    ent = _shannon_entropy(region)
    homo = _max_homopolymer(region)

    # score: spanning repeats dominate (mechanistic), span length scales risk,
    # low complexity adds; all terms normalized to rough 0..1 contributions
    s_pairs = min(len(spanning) / 20.0, 1.0)
    s_span = min(max_span / (2 * window), 1.0)
    s_complexity = 1.0 - min(ent / 6.0, 1.0)  # 10-mer entropy ~log2 scale
    s_homo = min(max(homo - 6, 0) / 10.0, 1.0)
    score = 0.45 * s_pairs + 0.30 * s_span + 0.15 * s_complexity + 0.10 * s_homo
    flag = "HIGH" if score >= 0.5 else ("MODERATE" if score >= 0.25 else "LOW")
    return RiskReport(
        n_repeat_pairs=len(spanning), max_deletion_span=max_span,
        left_repeat_density=ld, right_repeat_density=rd, asymmetry=asym,
        mean_entropy=ent, max_homopolymer=homo,
        score=round(score, 4), flag=flag,
        details={"window": window, "cut_pos": cut_pos,
                 "n_repeats_total": len(reps),
                 "subscores": {"pairs": round(s_pairs, 3), "span": round(s_span, 3),
                               "complexity": round(s_complexity, 3),
                               "homopolymer": round(s_homo, 3)}})
