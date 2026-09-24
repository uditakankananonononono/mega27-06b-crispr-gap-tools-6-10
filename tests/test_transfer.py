import numpy as np

from tools_transfer.divergence import js_divergence, species_shift
from tools_transfer.evaluator import quick_transfer_eval


def test_js_divergence_bounds():
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5])
    assert js_divergence(p, q) == 0.0
    r = np.array([1.0, 0.0])
    assert js_divergence(p, r) > 0.2


def test_species_shift_identical():
    seqs = ["ACGTACGTACGTACGTACGTACGTACGTAC"] * 10
    s = species_shift(seqs, seqs)
    assert s["gc_delta"] == 0.0
    assert s["kmer_js_divergence"] < 1e-9


def test_transfer_eval_synthetic():
    rng = np.random.default_rng(0)
    # species A: activity = GC-driven; species B: same rule
    def gen(n):
        ctx, y = [], []
        for _ in range(n):
            gc = rng.uniform(0.3, 0.7)
            p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
            s = "".join(rng.choice(list("ACGT"), size=30, p=p))
            ctx.append(s)
            y.append(gc)
        return ctx, np.array(y)
    ca, ya = gen(200)
    cb, yb = gen(80)
    v = quick_transfer_eval(ca, ya, cb, yb, "spA", "spB")
    assert -1.0 <= v.transfer_spearman <= 1.0
    assert isinstance(v.verdict, str) and v.verdict
