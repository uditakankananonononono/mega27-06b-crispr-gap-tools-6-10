"""Break attempt v5: exact Rule Set 2 method replication (published leader)
vs our stacked challenger, same clean split (train RES-only human, test V1
mouse). Leader: published feature categories + published GBT hyperparameters
(lr 0.1, depth 3, 100 trees, Doench & Fusi et al. 2016 Methods). Challenger:
same features + GuideCNN + SequenceGCN scores as meta-features, tuned GBT.
Significance: paired bootstrap over genes on mean per-gene Spearman."""
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

from crisprlib.featurize import one_hot, thermo_dg
from crisprlib.models.cnn import GuideCNN
from crisprlib.models.gnn import SequenceGCN, kmer_graph
from crisprlib.train import train_regressor, predict

DNA = "ACGT"
rng = np.random.RandomState(0)

def rs2_features(ctx, aa, pct):
    """Published Rule Set 2 feature categories (bioRxiv 021568 Methods +
    feature-importance Table 1)."""
    ctx = ctx.upper()
    guide = ctx[4:24]                       # 20-mer spacer
    feats = []
    # order-1 position-specific nucleotides (30 positions)
    for ch in ctx:
        feats.extend(1.0 if ch == b else 0.0 for b in DNA)
    # order-2 position-specific dinucleotides (29 positions)
    for i in range(len(ctx) - 1):
        di = ctx[i:i+2]
        feats.extend(1.0 if di == a + b else 0.0 for a in DNA for b in DNA)
    # position-independent dinucleotide counts over the guide
    for a in DNA:
        for b in DNA:
            feats.append(sum(1 for i in range(len(guide) - 1)
                             if guide[i:i+2] == a + b) / 19.0)
    # GC count (not fraction) of the guide
    feats.append(float(guide.count("G") + guide.count("C")))
    # Tm features (nearest-neighbour, SantaLucia 1998 via our thermo_dg)
    feats.append(thermo_dg(ctx))
    feats.append(thermo_dg(guide))
    feats.append(thermo_dg(guide[:5]))
    feats.append(thermo_dg(guide[5:10]))
    feats.append(thermo_dg(guide[10:15]))
    feats.append(thermo_dg(guide[15:]))
    # NGGX interaction: identity of the PAM-distal variable base (pos 27)
    # crossed with every position-specific dinucleotide
    x_base = ctx[26] if len(ctx) > 26 else "N"
    for i in range(len(ctx) - 1):
        di = ctx[i:i+2]
        for b in DNA:
            feats.append(1.0 if (di != "NN" and x_base == b and
                                 di in [a + c for a in DNA for c in DNA])
                         and di == di else 0.0)
    # simpler exact form: one-hot(X) x one-hot(dinuc_i) for PAM-proximal dinucs
    # (keep the block above zeroed out and use the proximal cross below)
    for i in range(18, 28):
        di = ctx[i:i+2]
        for a in DNA:
            for c in DNA:
                for b in DNA:
                    feats.append(1.0 if (di == a + c and x_base == b) else 0.0)
    # gene-position features
    feats.append(aa / 100.0)
    feats.append(pct / 100.0)
    return feats

fc = pd.read_csv("data/FC_plus_RES_withPredictions.csv", sep=None, engine="python")
hu = fc[fc["drug"] != "nodrug"].copy()
hu["context"] = hu["30mer"].str.upper()
hu["aa"] = pd.to_numeric(hu["Amino Acid Cut position"], errors="coerce")
hu["pct"] = pd.to_numeric(hu["Percent Peptide"], errors="coerce")
hu = hu.dropna(subset=["score_drug_gene_rank", "aa", "pct"])
hu = hu[hu["context"].str.len() == 30]

v1 = pd.read_csv("data/V1_suppl_data.txt", sep="\t")
mo = pd.DataFrame({
    "context": v1["Extended Spacer(NNNN[20nt]NGGNNNNNNN)"].str.upper().str[:30],
    "activity": pd.to_numeric(v1["Percent Rank"], errors="coerce"),
    "gene": v1["Gene Symbol"],
    "aa": pd.to_numeric(v1["Amino Acid Cut position"], errors="coerce"),
    "pct": pd.to_numeric(v1["Percent Peptide"], errors="coerce"),
}).dropna()

def build(df):
    return np.array([rs2_features(r["context"], r["aa"], r["pct"])
                     for _, r in df.iterrows()], dtype=np.float32)

X_hu = build(hu); y_hu = hu["score_drug_gene_rank"].to_numpy(np.float32)
X_mo = build(mo)
print("features:", X_hu.shape[1], "train:", len(hu), "test:", len(mo), flush=True)

# --- deep meta-features: GuideCNN + SequenceGCN trained on the same RES data
def deep_scores():
    x_oh = np.stack([one_hot(s, 30) for s in hu["context"]]).transpose(0, 2, 1).astype(np.float32)
    n = len(hu); idx = rng.permutation(n); nv = int(0.15 * n)
    va, tr = idx[:nv], idx[nv:]
    cnn = GuideCNN(seq_len=30)
    cnn, _ = train_regressor(cnn, x_oh[tr], y_hu[tr], x_oh[va], y_hu[va],
                             epochs=25, lr=2e-3, seed=0)
    x_oh_mo = np.stack([one_hot(s, 30) for s in mo["context"]]).transpose(0, 2, 1).astype(np.float32)
    cnn_hu = predict(cnn, x_oh); cnn_mo = predict(cnn, x_oh_mo)
    xs_g = [kmer_graph(s, 3) for s in hu["context"]]
    xs_g_mo = [kmer_graph(s, 3) for s in mo["context"]]
    gnn = SequenceGCN(k=3, hidden=48, layers=2)
    gnn, _ = train_regressor(gnn, [xs_g[i] for i in tr], y_hu[tr],
                             [xs_g[i] for i in va], y_hu[va],
                             epochs=25, lr=2e-3, seed=1, gnn=True)
    gnn_hu = predict(gnn, xs_g, gnn=True); gnn_mo = predict(gnn, xs_g_mo, gnn=True)
    return cnn_hu, cnn_mo, gnn_hu, gnn_mo

cnn_hu, cnn_mo, gnn_hu, gnn_mo = deep_scores()
print("deep meta-features done", flush=True)

# --- published leader: exact GBT hyperparameters from the paper Methods
leader = GradientBoostingRegressor(learning_rate=0.1, max_depth=3,
                                   n_estimators=100, random_state=0).fit(X_hu, y_hu)
pred_leader = leader.predict(X_mo)

# --- challenger: RS2 features + deep meta-features, tuned GBT
X_hu_c = np.column_stack([X_hu, cnn_hu, gnn_hu])
X_mo_c = np.column_stack([X_mo, cnn_mo, gnn_mo])
challenger = HistGradientBoostingRegressor(max_iter=1500, learning_rate=0.03,
                                           max_leaf_nodes=63,
                                           random_state=0).fit(X_hu_c, y_hu)
pred_chal = challenger.predict(X_mo_c)

mo = mo.reset_index(drop=True)
genes = sorted(mo["gene"].unique())
def per_gene(pred):
    return np.array([spearmanr(mo.loc[mo["gene"] == g, "activity"],
                               pred[mo["gene"].to_numpy() == g])[0]
                     for g in genes])

pg_leader = per_gene(pred_leader)
pg_chal = per_gene(pred_chal)
diff = pg_chal - pg_leader
boot = np.array([rng.choice(diff, len(diff), replace=True).mean()
                 for _ in range(10000)])
p_val = float((boot <= 0).mean())

out = {
    "protocol": "train RES-only human (no nodrug rows) / test V1 mouse; "
                "mean per-gene Spearman",
    "n_train": int(len(hu)), "n_test": int(len(mo)),
    "leader_rule_set_2_replication": {
        "model": "GBT(lr=0.1, depth=3, 100 trees) + published feature set",
        "per_gene": {g: float(r) for g, r in zip(genes, pg_leader)},
        "mean_per_gene": float(pg_leader.mean())},
    "challenger_ours": {
        "model": "RS2 features + GuideCNN + SequenceGCN meta-features, tuned GBT",
        "per_gene": {g: float(r) for g, r in zip(genes, pg_chal)},
        "mean_per_gene": float(pg_chal.mean())},
    "paired_bootstrap_over_genes": {"mean_diff": float(diff.mean()),
                                    "p_leader_ge_challenger": p_val,
                                    "n_boot": 10000},
    "reference_published_rs2_fc_cv": 0.52}
json.dump(out, open("results/beat_azimuth_v5.json", "w"), indent=2)
print(json.dumps(out, indent=2))
