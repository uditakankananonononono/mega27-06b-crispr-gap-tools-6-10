import numpy as np

from crisprlib.featurize import (align_pair, dinuc_one_hot, gc_content,
                                 kmer_counts, microhomology_score, one_hot,
                                 position_features, thermo_dg)


def test_one_hot_basic():
    x = one_hot("ACGT")
    assert x.shape == (4, 4)
    assert np.allclose(x.sum(axis=1), 1.0)
    assert x[0].tolist() == [1, 0, 0, 0]  # A


def test_one_hot_unknown_and_pad():
    x = one_hot("AN", length=4)
    assert x.shape == (4, 4)
    assert np.allclose(x[1], 0.0)
    assert np.allclose(x[2:], 0.0)


def test_dinuc():
    x = dinuc_one_hot("AAAA")
    assert x.shape == (3, 16)
    assert np.allclose(x[:, 0], 1.0)  # AA = index 0


def test_gc():
    assert gc_content("GGCC") == 1.0
    assert gc_content("ATAT") == 0.0
    assert gc_content("AGCT") == 0.5
    assert gc_content("") == 0.0


def test_thermo_stability_order():
    # GC-rich duplex is more stable (more negative dG)
    assert thermo_dg("GCGCGCGCGC") < thermo_dg("ATATATATAT")
    assert thermo_dg("A") == 0.0


def test_align_pair_mismatches():
    g, mm = align_pair("AAAA", "AAAT")
    assert mm.tolist() == [0.0, 0.0, 0.0, 1.0]
    assert g.shape == (4, 4)


def test_microhomology():
    s = microhomology_score("TTGCGCA", "GCGCATT")
    assert s > 0.0
    assert microhomology_score("AAAAAA", "TTTTTT") == 0.0


def test_kmer_counts():
    v = kmer_counts("AAAAAA", k=3)
    assert v.sum() == 4  # 4 overlapping AAA kmers
    assert v[0] == 4
