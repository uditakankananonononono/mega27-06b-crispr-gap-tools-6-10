import numpy as np

from crisprlib.benchmark import (classification_metrics, mit_aggregate,
                                 mit_score, regression_metrics)
from crisprlib.models import GuideCNN
from crisprlib.train import predict, train_regressor


def test_mit_score_identity_and_monotonic():
    g = "ACGTACGTACGTACGTACGT"
    assert mit_score(g, g) == 1.0
    one_mm = mit_score(g, "ACGTACGTACGTACGTACGA")
    many_mm = mit_score(g, "TTGTACGTTCGTACGTACGA")
    assert 0.0 <= many_mm < one_mm <= 1.0


def test_mit_score_seed_sensitivity():
    g = "ACGTACGTACGTACGTACGT"
    distal = mit_score(g, "TCGTACGTACGTACGTACGT")  # pos 1 mismatch
    seed = mit_score(g, "ACGTACGTACGTACGTACGA")    # pos 20 mismatch (seed)
    assert seed < distal  # seed mismatches penalized more


def test_mit_aggregate():
    g = "ACGTACGTACGTACGTACGT"
    agg = mit_aggregate(g, [g, "TTGTACGTTCGTACGTACGA"])
    assert 0.0 < agg < 100.0


def test_classification_metrics():
    m = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["auroc"] == 1.0 and m["auprc"] == 1.0


def test_regression_metrics_perfect():
    m = regression_metrics([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert abs(m["spearman"] - 1.0) < 1e-9
    assert abs(m["pearson"] - 1.0) < 1e-9


def test_train_regressor_learns():
    rng = np.random.default_rng(0)
    x = rng.random((80, 4, 12)).astype(np.float32)
    y = (x.sum(axis=(1, 2)) > x.sum(axis=(1, 2)).mean()).astype(np.float32)
    m = GuideCNN(seq_len=12, filters=8, kernel_widths=(3,))
    m, hist = train_regressor(m, x, y, epochs=60, batch=32, seed=1, lr=5e-3)
    p = predict(m, x)
    assert np.corrcoef(p, y)[0, 1] > 0.5
