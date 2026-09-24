"""Train + benchmark repair-outcome predictors on real inDelphi U2OS LibA data.

Targets: frameshift_frac, ins1_frac, mh_del_frac per site.
Models: CNN (one-hot context), GNN (k-mer transition graph).
Baselines: ridge on 3-mer counts. Published reference points:
- inDelphi (Shen 2018): per-genotype Pearson up to ~0.85 on held-out mESC.
- Lindel (Chen 2019): frameshift prediction R^2 ~0.85 on their test set.
We report honest held-out Spearman/Pearson for each model; no claim of parity
is made unless the numbers support it. Run live (not CI).
"""
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from crisprlib.featurize import (gc_content, kmer_counts, microhomology_score,
                                 one_hot, thermo_dg)
from crisprlib.models import GuideCNN, SequenceGCN, kmer_graph
from crisprlib.train import predict, train_regressor
from crisprlib.benchmark import regression_metrics
from tools_outcome.loader import aggregate_events, load_targets

tg = load_targets("data/targets-libA.txt")
sites = aggregate_events("data/U2OS_LibA_postCas9_rep1.csv", tg)
print(f"{len(sites)} sites")

TARGETS = ["frameshift_frac", "ins1_frac", "mh_del_frac"]
X_oh = np.stack([one_hot(s.context, 55) for s in sites]).transpose(0, 2, 1)
# positional feature matrix: per-position one-hot (flattened) + repair-relevant
# engineered features (microhomology across the cut, local GC, duplex dG)
def positional_features(ctx: str) -> np.ndarray:
    oh = one_hot(ctx, 55).reshape(-1)
    left, right = ctx[:28], ctx[28:]
    eng = np.array([microhomology_score(left, right), gc_content(ctx),
                    gc_content(left[-10:]), thermo_dg(left[-15:]),
                    thermo_dg(right[:15])], dtype=np.float32)
    return np.concatenate([oh, eng])
X_pos = np.stack([positional_features(s.context) for s in sites])
X_km = np.stack([kmer_counts(s.context, 3) for s in sites])
X_gr = [kmer_graph(s.context, k=3) for s in sites]

idx = np.arange(len(sites))
tr, te = train_test_split(idx, test_size=0.2, random_state=0)
res = {"n_train": int(len(tr)), "n_test": int(len(te)), "targets": {}}
for tname in TARGETS:
    y = np.array([getattr(s, tname) for s in sites], dtype=np.float32)
    out = {}
    ridge = Ridge(alpha=1.0).fit(X_km[tr], y[tr])
    out["ridge_kmer"] = regression_metrics(y[te], ridge.predict(X_km[te]))
    gbt = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                        random_state=0).fit(X_pos[tr], y[tr])
    out["gbt_positional"] = regression_metrics(y[te], gbt.predict(X_pos[te]))
    cnn = GuideCNN(seq_len=55, filters=24, kernel_widths=(3, 5, 7))
    cnn, _ = train_regressor(cnn, X_oh[tr], y[tr], X_oh[te], y[te], epochs=80, lr=3e-3, seed=0)
    out["cnn"] = regression_metrics(y[te], predict(cnn, X_oh[te]))
    gnn = SequenceGCN(k=3, hidden=48, layers=2)
    gnn, _ = train_regressor(gnn, [X_gr[i] for i in tr], y[tr],
                             [X_gr[i] for i in te], y[te],
                             epochs=80, lr=3e-3, seed=0, gnn=True)
    out["gnn"] = regression_metrics(y[te], predict(gnn, [X_gr[i] for i in te], gnn=True))
    res["targets"][tname] = out
    print(tname, json.dumps(out))
json.dump(res, open("results/outcome_models.json", "w"), indent=2)
