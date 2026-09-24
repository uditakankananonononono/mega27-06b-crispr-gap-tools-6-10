"""GNN components: sequence-as-graph scoring with a pure-torch GCN.

A sequence is represented as a k-mer transition graph: nodes are k-mers, edges
follow consecutive k-mer overlaps. Node features are k-mer composition; the GCN
aggregates neighbourhood structure that a CNN over the linear sequence cannot
see directly (repeated motifs, palindromic structure).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..featurize import DNA_ALPHABET, kmer_counts


def _kmers(seq: str, k: int):
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


def kmer_graph(seq: str, k: int = 3, max_nodes: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Build (node_features, adjacency) for the k-mer transition graph.

    Nodes: up to max_nodes distinct k-mers (first occurrence order).
    Node features: (4k,) one-hot composition of the k-mer.
    Adjacency: symmetrized, self-loops added, row-normalized (D^-1/2 A D^-1/2
    up to a constant); float32 (N, N).
    """
    seq = seq.upper()
    km = [s for s in _kmers(seq, k) if all(b in DNA_ALPHABET for b in s)]
    nodes: list[str] = []
    index: dict[str, int] = {}
    for s in km:
        if s not in index and len(nodes) < max_nodes:
            index[s] = len(nodes)
            nodes.append(s)
    n = len(nodes)
    x = np.zeros((n, 4 * k), dtype=np.float32)
    base_idx = {b: i for i, b in enumerate(DNA_ALPHABET)}
    for i, s in enumerate(nodes):
        for pos, b in enumerate(s):
            x[i, pos * 4 + base_idx[b]] = 1.0
    a = np.eye(n, dtype=np.float32)
    for s1, s2 in zip(km, km[1:]):
        i, j = index.get(s1), index.get(s2)
        if i is not None and j is not None:
            a[i, j] = 1.0
            a[j, i] = 1.0
    if n == 0:
        return np.zeros((1, 4 * k), dtype=np.float32), np.ones((1, 1), dtype=np.float32)
    deg = a.sum(axis=1)
    dinv = np.power(np.maximum(deg, 1e-6), -0.5)
    a_norm = (a * dinv[:, None]) * dinv[None, :]
    return x, a_norm.astype(np.float32)


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.bmm(a, self.lin(x)))


class SequenceGCN(nn.Module):
    """GCN over k-mer transition graphs with global mean pooling."""

    def __init__(self, k: int = 3, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.2, binary: bool = False):
        super().__init__()
        self.k = k
        in_dim = 4 * k
        dims = [in_dim] + [hidden] * layers
        self.layers = nn.ModuleList([GCNLayer(dims[i], dims[i + 1]) for i in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
        self.binary = binary

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = self.dropout(layer(h, a))
        pooled = h.mean(dim=1)
        out = self.head(pooled).squeeze(-1)
        return torch.sigmoid(out) if self.binary else out
