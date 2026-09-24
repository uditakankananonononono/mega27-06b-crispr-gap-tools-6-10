"""Multiplex gRNA interference screen.

Two interference channels, each scored per ordered guide pair:

1. Duplex interference: guide RNAs co-expressed in one cell bind each other
   (spacer-spacer trans-duplexes) and self-fold, sequestering guides from
   Cas9. Channel metrics: ViennaRNA heteroduplex MFE, cofold MFE, self-fold
   MFE, contiguous spacer complementarity.

2. PAM/cut-site competition: guides cutting within a shared repair window
   (~50 bp) compete for the same repair machinery and generate inter-cut
   deletions that are usually unintended in multiplex knock-in designs.
   Channel metric: cut-site distance -> competition weight exp(-d / lambda).

Aggregate per-guide interference score and a global multiplex compatibility
score in [0, 1] (1 = no detectable interference).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .duplex import (cofold_mfe, heteroduplex_mfe, self_fold_mfe,
                     spacer_complementarity)

REPAIR_WINDOW_BP = 50
DUPLEX_STRONG_KCAL = -12.0   # ~6 contiguous bp of perfect duplex at 37C
SELF_FOLD_STRONG_KCAL = -6.0


@dataclass
class Guide:
    guide_id: str
    spacer: str                # 20-nt protospacer (5'->3')
    pam: str = "NGG"
    cut_pos: int | None = None  # genomic cut coordinate (same locus scale)
    chrom: str | None = None

    def __post_init__(self):
        self.spacer = self.spacer.upper()


@dataclass
class PairReport:
    guide_a: str
    guide_b: str
    heteroduplex_mfe: float
    cofold_mfe: float
    complementarity: float
    cut_distance_bp: int | None
    pam_competition: float
    duplex_flag: bool
    pam_flag: bool


@dataclass
class ScreenResult:
    pair_reports: list[PairReport] = field(default_factory=list)
    per_guide: dict = field(default_factory=dict)  # id -> {score, flags}
    compatibility: float = 1.0


def pam_competition_weight(distance_bp: int | None, lam: float = REPAIR_WINDOW_BP) -> float:
    """Exponential decay of repair-machinery competition with cut distance."""
    if distance_bp is None:
        return 0.0
    return math.exp(-abs(distance_bp) / lam)


class MultiplexScreener:
    def __init__(self, lam: float = REPAIR_WINDOW_BP,
                 duplex_kcal: float = DUPLEX_STRONG_KCAL,
                 self_kcal: float = SELF_FOLD_STRONG_KCAL):
        self.lam = lam
        self.duplex_kcal = duplex_kcal
        self.self_kcal = self_kcal

    def score_pair(self, a: Guide, b: Guide) -> PairReport:
        hd = heteroduplex_mfe(a.spacer, b.spacer)
        cf = cofold_mfe(a.spacer, b.spacer)
        comp = spacer_complementarity(a.spacer, b.spacer)
        dist = None
        if a.cut_pos is not None and b.cut_pos is not None and a.chrom == b.chrom:
            dist = abs(a.cut_pos - b.cut_pos)
        comp_w = pam_competition_weight(dist, self.lam)
        return PairReport(
            guide_a=a.guide_id, guide_b=b.guide_id,
            heteroduplex_mfe=hd, cofold_mfe=cf, complementarity=comp,
            cut_distance_bp=dist, pam_competition=comp_w,
            duplex_flag=(hd <= self.duplex_kcal) or (comp >= 0.4),
            pam_flag=(dist is not None and dist <= self.lam),
        )

    def screen(self, guides: list[Guide]) -> ScreenResult:
        res = ScreenResult()
        n = len(guides)
        for i in range(n):
            for j in range(i + 1, n):
                res.pair_reports.append(self.score_pair(guides[i], guides[j]))
        # per-guide aggregation
        for g in guides:
            sf = self_fold_mfe(g.spacer)
            pairs = [p for p in res.pair_reports
                     if g.guide_id in (p.guide_a, p.guide_b)]
            dup_hits = sum(p.duplex_flag for p in pairs)
            pam_hits = sum(p.pam_flag for p in pairs)
            worst_duplex = min((p.heteroduplex_mfe for p in pairs), default=0.0)
            # per-guide penalty: strong self-fold, each flagged pair
            penalty = (1.0 if sf <= self.self_kcal else 0.0) + dup_hits + pam_hits
            res.per_guide[g.guide_id] = {
                "self_fold_mfe": sf,
                "self_fold_flag": sf <= self.self_kcal,
                "duplex_flagged_pairs": dup_hits,
                "pam_flagged_pairs": pam_hits,
                "worst_duplex_mfe": worst_duplex,
                "interference_penalty": penalty,
            }
        total_pairs = max(len(res.pair_reports), 1)
        flagged = sum(1 for p in res.pair_reports if p.duplex_flag or p.pam_flag)
        res.compatibility = 1.0 - flagged / total_pairs
        return res


def screen_guides(spacers: list[str], cut_pos: list[int] | None = None,
                  chrom: str | None = None) -> ScreenResult:
    guides = []
    for i, sp in enumerate(spacers):
        guides.append(Guide(guide_id=f"g{i + 1}", spacer=sp,
                            cut_pos=(cut_pos[i] if cut_pos else None),
                            chrom=chrom))
    return MultiplexScreener().screen(guides)
