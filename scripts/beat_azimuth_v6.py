"""Break attempt v6: fixes both v5 flaws. (1) Leak-free stacking: CNN/GNN
meta-features for training rows are out-of-fold (3-fold) predictions; test
meta-features come from final models trained on all RES rows. (2) The
meta-learner uses the published conservative hyperparameters (GBT lr=0.1,
depth 3, 100 trees), same as the leader, so the only difference is the added
deep meta-features. Leader re-run here for an identical-seed comparison."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

from crisprlib.featurize import one_hot, thermo_dg
from crisprlib.models.cnn import GuideCNN
from crisprlib.models.gnn import SequenceGCN, kmer_graph
from crisprlib.train import train_regressor, predict

DNA = "ACGT"
rng = np.random.RandomState(0)

def rs2_features(ctx, aa, pct):
    ctx = ctx.upper(); guide = ctx[4:24]; feats = []
    for ch in ctx:
        feats.extend(1.0 if ch == b else 0.0 for b in DNA)
    for i in range(len(ctx) - 1):
        di = ctx[i:i+2]
        feats.extend(1.0 if di == a + b else 0.0 for a in DNA for b in DNA)
    for a in DNA:
        for b in DNA:
            feats.append(sum(1 for i in range(len(guide) - 1)
                             if guide[i:i+2] == a + b) / 19.0)
    feats.append(float(guide.count("G") + guide.count("C")))
    feats.append(thermo_dg(ctx)); feats.append(thermo_dg(guide))
    feats.append(thermo_dg(guide[:5])); feats.append(thermo_dg(guide[5:10]))
    feats.append(thermo_dg(guide[10:15])); feats.append(thermo_dg(guide[15:]))
    x_base = ctx[26] if len(ctx) > 26 else "N"
    for i in range(18, 28):
        di = ctx[i:i+2]
        for a in DNA:
            for c in DNA:
                for b in DNA:
                    feats.append(1.0 if (di == a + c and x_base == b) else 0.0)
    feats.append(aa / 100.0); feats.append(pct / 100.0)
    return feats

fc = pd.read_csv("data/FC_plus_RES_withPredictions.csv", sep=None, engine="python")
hu = fc[fc["drug"] != "nodrug"].copy()
hu["context"] = hu["30mer"].str.upper()
hu["aa"] = pd.to_numeric(hu["Amino Acid Cut position"], errors="coerce")
hu["pct"] = pd.to_numeric(hu["Percent Peptide"], errors="coerce")
hu = hu.dropna(subset=["score_drug_gene_rank", "aa", "pct"])
hu = hu[hu["context"].str.len() == 30].reset_index(drop=True)

v1 = pd.read_csv("data/V1_suppl_data.txt", sep="\t")
mo = pd.DataFrame({
    "context": v1["Extended Spacer(NNNN[20nt]NGGNNNNNNN)"].str.upper().str[:30],
    "activity": pd.to_numeric(v1["Percent Rank"], errors="coerce"),
    "gene": v1["Gene Symbol"],
    "aa": pd.to_numeric(v1["Amino Acid Cut position"], errors="coerce"),
    "pct": pd.to_numeric(v1["Percent Peptide"], errors="coerce"),
}).dropna().reset_index(drop=True)

def build(df):
    return np.array([rs2_features(r["context"], r["aa"], r["pct"])
                     for _, r in df.iterrows()], dtype=np.float32)

X_hu = build(hu); y_hu = hu["score_drug_gene_rank"].to_numpy(np.float32)
X_mo = build(mo)
print("features:", X_hu.shape[1], flush=True)

x_oh_hu = np.stack([one_hot(s, 30) for s in hu["context"]]).transpose(0, 2, 1).astype(np.float32)
x_oh_mo = np.stack([one_hot(s, 30) for s in mo["context"]]).transpose(0, 2, 1).astype(np.float32)
G_hu = [kmer_graph(s, k=3) for s in hu["context"]]
G_mo = [kmer_graph(s, k=3) for s in mo["context"]]

def train_deep(tr_idx, seed):
    n = len(tr_idx); sub = rng.permutation(n); nv = int(0.15 * n)
    va_i, tr_i = tr_idx[sub[:nv]], tr_idx[sub[nv:]]
    cnn = GuideCNN(seq_len=30)
    cnn, _ = train_regressor(cnn, x_oh_hu[tr_i], y_hu[tr_i],
                             x_oh_hu[va_i], y_hu[va_i], epochs=25, lr=2e-3, seed=seed)
    gnn = SequenceGCN(k=3, hidden=48, layers=2)
    gnn, _ = train_regressor(gnn, [G_hu[i] for i in tr_i], y_hu[tr_i],
                             [G_hu[i] for i in va_i], y_hu[va_i],
                             epochs=25, lr=2e-3, seed=seed + 1, gnn=True)
    return cnn, gnn

# out-of-fold deep scores for training rows (leak-free stacking)
oof_cnn = np.zeros(len(hu), dtype=np.float32)
oof_gnn = np.zeros(len(hu), dtype=np.float32)
kf = KFold(n_splits=3, shuffle=True, random_state=0)
for fold, (tr_i, te_i) in enumerate(kf.split(X_hu)):
    cnn, gnn = train_deep(tr_i, seed=10 * fold)
    oof_cnn[te_i] = predict(cnn, x_oh_hu[te_i])
    oof_gnn[te_i] = predict(gnn, [G_hu[i] for i in te_i], gnn=True)
    print(f"fold {fold} oof done", flush=True)

# final deep models on all RES rows for test predictions
cnn_f, gnn_f = train_deep(np.arange(len(hu)), seed=99)
mo_cnn = predict(cnn_f, x_oh_mo)
mo_gnn = predict(gnn_f, G_mo, gnn=True)
print("final deep models done", flush=True)

leader = GradientBoostingRegressor(learning_rate=0.1, max_depth=3,
                                   n_estimators=100, random_state=0).fit(X_hu, y_hu)
pred_leader = leader.predict(X_mo)

X_hu_s = np.column_stack([X_hu, oof_cnn, oof_gnn])
X_mo_s = np.column_stack([X_mo, mo_cnn, mo_gnn])
chal = GradientBoostingRegressor(learning_rate=0.1, max_depth=3,
                                 n_estimators=100, random_state=0).fit(X_hu_s, y_hu)
pred_chal = chal.predict(X_mo_s)

genes = sorted(mo["gene"].unique())
def per_gene(pred):
    return np.array([spearmanr(mo.loc[mo["gene"] == g, "activity"],
                               pred[mo["gene"].to_numpy() == g])[0]
                     for g in genes])
pg_l, pg_c = per_gene(pred_leader), per_gene(pred_chal)
diff = pg_c - pg_l
boot = np.array([rng.choice(diff, len(diff), replace=True).mean()
                 for _ in range(10000)])
out = {
    "protocol": "train RES-only human / test V1 mouse; mean per-gene Spearman; "
                "leak-free 3-fold OOF deep meta-features; identical published "
                "GBT hyperparameters for leader and challenger",
    "leader_rule_set_2_replication": {
        "per_gene": {g: float(r) for g, r in zip(genes, pg_l)},
        "mean_per_gene": float(pg_l.mean())},
    "challenger_rs2_plus_deep_oof": {
        "per_gene": {g: float(r) for g, r in zip(genes, pg_c)},
        "mean_per_gene": float(pg_c.mean())},
    "paired_bootstrap_over_genes": {"mean_diff": float(diff.mean()),
                                    "p_leader_ge_challenger": float((boot <= 0).mean()),
                                    "n_boot": 10000}}
json.dump(out, open("results/beat_azimuth_v6.json", "w"), indent=2)
print(json.dumps(out, indent=2))
