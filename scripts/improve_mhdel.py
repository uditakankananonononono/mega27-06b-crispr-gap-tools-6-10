"""Beat attempt: mh_del_frac prediction with mechanistic MH-pair enumeration.

Current standing (results/outcome_models.json): GBT 0.173 Spearman, ridge ~?
Hypothesis: per-deletion microhomology enumeration (the feature family inDelphi
uses for its MH-deletion genotype model) lifts the aggregate fraction model.
Honest gate: report held-out Spearman vs the existing GBT and ridge baselines.
"""
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from crisprlib.featurize import gc_content, kmer_counts, one_hot
from crisprlib.benchmark import regression_metrics
from tools_outcome.loader import aggregate_events, load_targets

CUT = 28
WIN = 20
MAXDEL = 40

def mh_pair_features(ctx: str) -> np.ndarray:
    """Enumerate deletions within cut +/- WIN, length 1..MAXDEL, that span the
    cut; for each, microhomology arm length = longest k with
    ctx[i:i+k] == ctx[j:j+k]. Aggregate inDelphi-style."""
    mh_lens, del_lens, scores = [], [], []
    for i in range(max(0, CUT - WIN), CUT + 1):
        for L in range(1, MAXDEL + 1):
            j = i + L
            if j <= CUT or j >= len(ctx):
                continue
            k = 0
            while j + k < len(ctx) and i + k < len(ctx) and ctx[i + k] == ctx[j + k] and k < 12:
                k += 1
            if k >= 2:
                mh_lens.append(k); del_lens.append(L)
                scores.append(np.exp(0.6 * k - 0.15 * L))
    if not mh_lens:
        return np.zeros(8, dtype=np.float32)
    mh = np.array(mh_lens); dl = np.array(del_lens); sc = np.array(scores)
    return np.array([
        mh.max(), (mh >= 3).sum(), (mh >= 4).sum(),
        sc.sum(), sc.max(),
        (mh / dl).max(),
        gc_content(ctx[CUT-15:CUT+15]),
        np.log1p(len(mh)),
    ], dtype=np.float32)

tg = load_targets("data/targets-libA.txt")
sites = aggregate_events("data/U2OS_LibA_postCas9_rep1.csv", tg)
y = np.array([s.mh_del_frac for s in sites], dtype=np.float32)

X_mh = np.stack([mh_pair_features(s.context) for s in sites])
X_km = np.stack([kmer_counts(s.context, 3) for s in sites])
X_oh = np.stack([one_hot(s.context, 55).reshape(-1) for s in sites])
X_full = np.concatenate([X_oh, X_km, X_mh], axis=1)

idx = np.arange(len(sites))
tr, te = train_test_split(idx, test_size=0.2, random_state=0)
res = {}
res["ridge_kmer"] = regression_metrics(y[te], Ridge(alpha=1.0).fit(X_km[tr], y[tr]).predict(X_km[te]))
res["gbt_mh_only"] = regression_metrics(y[te], HistGradientBoostingRegressor(
    max_iter=600, learning_rate=0.05, random_state=0).fit(X_mh[tr], y[tr]).predict(X_mh[te]))
res["gbt_full_with_mh"] = regression_metrics(y[te], HistGradientBoostingRegressor(
    max_iter=600, learning_rate=0.05, random_state=0).fit(X_full[tr], y[tr]).predict(X_full[te]))
res["reference_existing_gbt_spearman"] = 0.173
print(json.dumps(res, indent=2))
with open("results/mhdel_improve.json", "w") as f:
    json.dump(res, f, indent=2)

# --- attempt 2: tuned GBT on the original positional feature set
from crisprlib.featurize import microhomology_score, thermo_dg

def positional_features(ctx: str) -> np.ndarray:
    oh = one_hot(ctx, 55).reshape(-1)
    left, right = ctx[:28], ctx[28:]
    feats = [microhomology_score(left, right), gc_content(ctx),
             gc_content(left[-10:]), thermo_dg(left[-15:]),
             thermo_dg(right[:15])]
    for d in (2, 3, 4, 5, 7, 10, 15):
        if len(left) > d and len(right) > d:
            feats.append(microhomology_score(left[:-d], right[d:]))
        else:
            feats.append(0.0)
    return np.concatenate([oh, np.array(feats, dtype=np.float32)])

X_pos = np.stack([positional_features(s.context) for s in sites])
res2 = {}
res2["gbt_tuned_pos"] = regression_metrics(y[te], HistGradientBoostingRegressor(
    max_iter=1500, learning_rate=0.03, max_leaf_nodes=63, random_state=0
    ).fit(X_pos[tr], y[tr]).predict(X_pos[te]))
res2["gbt_tuned_pos_mh"] = regression_metrics(y[te], HistGradientBoostingRegressor(
    max_iter=1500, learning_rate=0.03, max_leaf_nodes=63, random_state=0
    ).fit(np.concatenate([X_pos, X_mh], axis=1)[tr], y[tr]).predict(
        np.concatenate([X_pos, X_mh], axis=1)[te]))
res["attempt2_tuned"] = res2
print(json.dumps(res2, indent=2))
with open("results/mhdel_improve.json", "w") as f:
    json.dump(res, f, indent=2)
