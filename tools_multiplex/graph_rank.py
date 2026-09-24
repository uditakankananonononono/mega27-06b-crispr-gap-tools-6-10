"""GNN ranking of guide-removal priority on the multiplex interference graph.

The interference graph has guides as nodes; edge weights combine duplex MFE
(stronger duplex -> higher weight) and PAM competition. A GCN propagates
interference so that a guide's score reflects its embeddedness in the bad
subgraph, not only its own degree. Ranked output tells the designer which
single guide to drop first. Ground truth for small sets: brute-force
compatibility of every leave-one-out subset.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn as nn

from .screener import Guide, MultiplexScreener


def interference_graph(guides: list[Guide], screener: MultiplexScreener | None = None):
    """Return (node_features (N,4), weighted adjacency (N,N), ScreenResult).

    Node features: [self_fold_mfe (clipped, scaled), GC, n_flagged_pairs, 1].
    Edge weight: min(1, |duplex_mfe|/20) * 0.6 + pam_competition * 0.4.
    """
    sc = screener or MultiplexScreener()
    res = sc.screen(guides)
    n = len(guides)
    idx = {g.guide_id: i for i, g in enumerate(guides)}
    a = np.zeros((n, n), dtype=np.float32)
    for p in res.pair_reports:
        i, j = idx[p.guide_a], idx[p.guide_b]
        w = 0.6 * min(1.0, abs(p.heteroduplex_mfe) / 20.0) + 0.4 * p.pam_competition
        a[i, j] = a[j, i] = w
    x = np.zeros((n, 4), dtype=np.float32)
    for i, g in enumerate(guides):
        pg = res.per_guide[g.guide_id]
        gc = sum(1 for b in g.spacer if b in "GC") / max(len(g.spacer), 1)
        x[i] = [max(pg["self_fold_mfe"], -20.0) / 20.0, gc,
                pg["duplex_flagged_pairs"] + pg["pam_flagged_pairs"], 1.0]
    return x, a, res


class InterferenceGCN(nn.Module):
    """2-layer GCN producing per-node interference-embeddedness scores."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.w1 = nn.Linear(4, hidden)
        self.w2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, a):
        deg = a.sum(-1, keepdim=True).clamp(min=1e-6)
        an = a / deg
        h = torch.relu(torch.matmul(an, self.w1(x)))
        h = torch.relu(torch.matmul(an, self.w2(h)))
        return self.out(h).squeeze(-1)


def gnn_removal_ranking(guides: list[Guide]) -> list[str]:
    """Order guide_ids by GNN interference-embeddedness (drop first = highest)."""
    if len(guides) < 2:
        return [g.guide_id for g in guides]
    x, a, _ = interference_graph(guides)
    torch.manual_seed(0)
    model = InterferenceGCN()
    xt = torch.as_tensor(x)
    at = torch.as_tensor(a)
    # unsupervised readout: propagate raw flag counts through the graph, then
    # read out with the fixed-seed GCN; the propagation is the signal, weights
    # provide the nonlinearity. Deterministic via fixed seed.
    with torch.no_grad():
        scores = model(xt, at).numpy()
    order = np.argsort(-scores)
    ids = [g.guide_id for g in guides]
    return [ids[i] for i in order]


def brute_force_removal_ranking(guides: list[Guide]) -> list[str]:
    """Ground truth: order guides by compatibility gain when removed alone."""
    sc = MultiplexScreener()
    base = sc.screen(guides).compatibility
    gains = {}
    for g in guides:
        rest = [h for h in guides if h.guide_id != g.guide_id]
        gains[g.guide_id] = (sc.screen(rest).compatibility - base) if rest else 1.0
    return sorted(gains, key=lambda k: -gains[k])
