import numpy as np
import torch

from crisprlib.models import GuideCNN, PairCNN, SequenceGCN, kmer_graph


def test_guidecnn_forward():
    m = GuideCNN(seq_len=23)
    x = torch.rand(8, 4, 23)
    y = m(x)
    assert y.shape == (8,)


def test_paircnn_forward():
    m = PairCNN(seq_len=23)
    x = torch.rand(8, 9, 23)
    y = m(x)
    assert y.shape == (8,)
    assert ((y >= 0) & (y <= 1)).all()


def test_kmer_graph_shapes():
    x, a = kmer_graph("ACGTACGTACGT", k=3)
    assert x.shape[0] == a.shape[0] == a.shape[1]
    assert x.shape[1] == 12
    # row-normalized adjacency (symmetric, spectral radius <= 1)
    assert np.allclose(a, a.T)


def test_kmer_graph_max_nodes():
    x, a = kmer_graph("A" * 200, k=3, max_nodes=10)
    assert x.shape[0] <= 10


def test_sequencegcn_forward():
    m = SequenceGCN(k=3)
    xs, adjs = [], []
    for _ in range(4):
        x, a = kmer_graph("ACGTACGTACGTACGT", k=3)
        xs.append(x)
        adjs.append(a)
    xt = torch.as_tensor(np.stack(xs))
    at = torch.as_tensor(np.stack(adjs))
    y = m(xt, at)
    assert y.shape == (4,)
