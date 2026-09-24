import numpy as np

from tools_multiplex.duplex import (cofold_mfe, heteroduplex_mfe,
                                    self_fold_mfe, spacer_complementarity)
from tools_multiplex.graph_rank import (brute_force_removal_ranking,
                                        gnn_removal_ranking,
                                        interference_graph)
from tools_multiplex.screener import (Guide, MultiplexScreener,
                                      pam_competition_weight, screen_guides)


def test_perfect_complements_duplex_strong():
    e = heteroduplex_mfe("ACGUCGACGUACGUCGACGU", "ACGUCGACGUACGUCGACGU")
    r = heteroduplex_mfe("AAAAAAAAAAAAAAAAAAAA", "AAAAAAAAAAAAAAAAAAAA")
    assert e < r  # complementary pair binds more strongly


def test_self_fold_hairpin():
    hp = self_fold_mfe("GCGCGCAAAGCGCGC")
    assert hp < 0.0  # real hairpin has negative MFE


def test_complementarity_perfect():
    # a spacer vs its own reverse complement pairs end to end
    assert spacer_complementarity("ACGUACGUACGUACGUACGU", "ACGUACGUACGUACGUACGU") == 1.0
    # homopolymer-A spacer vs homopolymer-A spacer: no pairing at any offset
    assert spacer_complementarity("AAAAAAAAAAAAAAAAAAAA", "AAAAAAAAAAAAAAAAAAAA") == 0.0


def test_pam_weight_decay():
    assert pam_competition_weight(0) == 1.0
    assert pam_competition_weight(50) < pam_competition_weight(10)
    assert pam_competition_weight(None) == 0.0


def test_screener_flags_identical_guides():
    res = screen_guides(["ACGUACGUACGUACGUACGU"] * 3)
    assert all(p.duplex_flag for p in res.pair_reports)
    assert res.compatibility == 0.0


def test_screener_pam_competition():
    g = [Guide("g1", "ACGUACGUACGUACGUACGU", cut_pos=1000, chrom="chr1"),
         Guide("g2", "UUUUAAAACCCCGGGGAAAA", cut_pos=1020, chrom="chr1"),
         Guide("g3", "GGGGCCCCAAAATTTTGGGG", cut_pos=90000, chrom="chr1")]
    res = MultiplexScreener().screen(g)
    near = [p for p in res.pair_reports if p.cut_distance_bp == 20][0]
    far = [p for p in res.pair_reports if p.cut_distance_bp == 89000][0]
    assert near.pam_flag and not far.pam_flag
    assert near.pam_competition > far.pam_competition


def test_graph_shapes():
    g = [Guide(f"g{i}", s) for i, s in enumerate(
        ["ACGUACGUACGUACGUACGU", "ACGUACGUACGUACGUACGU",
         "GGGGCCCCAAAATTTTGGGG", "TTTTGGGGCCCCAAAAUUUU"])]
    x, a, res = interference_graph(g)
    assert x.shape == (4, 4) and a.shape == (4, 4)
    assert np.allclose(a, a.T)


def test_gnn_ranking_finds_bad_guide():
    # three mutually-duplexing identical guides + one orthogonal guide
    good = "GCGCGCGCGCGCGCGCGCGC"
    dup = "ACGUACGUACGUACGUACGU"
    guides = [Guide("bad1", dup), Guide("bad2", dup), Guide("bad3", dup),
              Guide("good", good)]
    bf = brute_force_removal_ranking(guides)
    assert bf[0] in ("bad1", "bad2", "bad3")
    gnn = gnn_removal_ranking(guides)
    assert gnn[0] in ("bad1", "bad2", "bad3")
