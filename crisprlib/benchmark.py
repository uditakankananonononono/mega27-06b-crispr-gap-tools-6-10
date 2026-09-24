"""Benchmark metrics + published-baseline scorers used across the gap tools."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import average_precision_score, roc_auc_score


def classification_metrics(y_true, p_score) -> dict:
    y_true = np.asarray(y_true).astype(int)
    p_score = np.asarray(p_score, dtype=float)
    out = {}
    if len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, p_score))
        out["auprc"] = float(average_precision_score(y_true, p_score))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rho, _ = spearmanr(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    return {"spearman": float(rho), "pearson": float(r)}


def mit_score(guide: str, offtarget: str) -> float:
    """MIT specificity score (Hsu et al. 2013, Nat Biotechnol 31:827).

    score = prod(1 - w_i) * (1 / (((19 - mean_pairwise_dist) / 19) * 4 + 1)) * 1/n_mm^2
    with the published per-position weights w. Exact reimplementation of the
    crispr.mit.edu formula; used as a baseline for off-target ranking.
    """
    W = [0.0, 0.0, 0.014, 0.0, 0.0, 0.395, 0.317, 0.0, 0.389, 0.079,
         0.445, 0.508, 0.613, 0.851, 0.732, 0.828, 0.615, 0.804,
         0.685, 0.583]
    guide = guide.upper()
    offtarget = offtarget.upper()
    n = min(len(guide), len(offtarget), 20)
    mm = [i for i in range(n) if guide[i] != offtarget[i]]
    if not mm:
        return 1.0
    score = 1.0
    for i in mm:
        score *= (1.0 - W[i])
    if len(mm) > 1:
        dists = [abs(a - b) for x, a in enumerate(mm) for b in mm[x + 1:]]
        d = float(np.mean(dists))
    else:
        d = 19.0
    score *= 1.0 / (((19.0 - d) / 19.0) * 4.0 + 1.0)
    score *= 1.0 / (len(mm) ** 2)
    return score


def mit_aggregate(guide: str, offtargets: list[str]) -> float:
    """Aggregate guide-level specificity: 100 / (100 + sum of site scores)."""
    return 100.0 / (100.0 + sum(mit_score(guide, ot) for ot in offtargets))
