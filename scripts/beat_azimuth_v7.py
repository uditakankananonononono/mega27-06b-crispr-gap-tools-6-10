"""Break attempt v7: multi-assay training. Identical RS2 features and
identical GBT hyperparameters (lr=0.1, depth=3, 100 trees) as the leader
replication; the ONLY change is the training set:
(a) leader: RES human only -> replicate the 0.569 bar
(b) challenger: RES + Doench V2 human (4,195 A375 guides, 15 genes)
Test: V1 mouse, mean per-gene Spearman. Same protocol as the whole ladder.
V2 parsed from Azimuth repo V2_data.xlsx ResultsFiltered sheet (real data).
"""
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, "scripts")
from _rs2feat import rs2_features

fc = pd.read_csv("data/FC_plus_RES_withPredictions.csv", sep=None, engine="python")
hu = fc[fc["drug"] != "nodrug"].copy()
hu["context"] = hu["30mer"].str.upper()
hu["aa"] = pd.to_numeric(hu["Amino Acid Cut position"], errors="coerce")
hu["pct"] = pd.to_numeric(hu["Percent Peptide"], errors="coerce")
hu = hu.dropna(subset=["score_drug_gene_rank", "aa", "pct"])
hu = hu[hu["context"].str.len() == 30].reset_index(drop=True)

v2 = pd.read_csv("data/V2_results.csv").dropna().reset_index(drop=True)

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
X_v2 = build(v2); y_v2 = v2["score"].to_numpy(np.float32)
X_mo = build(mo)
print("shapes", X_hu.shape, X_v2.shape, X_mo.shape, flush=True)

def per_gene_spearman(pred):
    vals = []
    for g, sub in mo.groupby("gene"):
        if len(sub) < 5:
            continue
        r = spearmanr(sub["activity"], pred[sub.index])[0]
        if np.isfinite(r):
            vals.append(r)
    return float(np.mean(vals)), len(vals)

def gbt():
    return GradientBoostingRegressor(learning_rate=0.1, max_depth=3,
                                     n_estimators=100, random_state=0)

res = {}
m = gbt().fit(X_hu, y_hu)
res["leader_res_only"] = per_gene_spearman(m.predict(X_mo))
print("leader", res["leader_res_only"], flush=True)

X_all = np.vstack([X_hu, X_v2]); y_all = np.concatenate([y_hu, y_v2])
m = gbt().fit(X_all, y_all)
res["challenger_res_plus_v2"] = per_gene_spearman(m.predict(X_mo))
print("challenger", res["challenger_res_plus_v2"], flush=True)

# bootstrap gene-level resampling for the paired difference
rng = np.random.RandomState(0)
genes = [g for g, sub in mo.groupby("gene") if len(sub) >= 5]
m1 = gbt().fit(X_hu, y_hu).predict(X_mo)
m2 = gbt().fit(X_all, y_all).predict(X_mo)
per_g = {}
for g in genes:
    sub = mo[mo["gene"] == g]
    per_g[g] = (spearmanr(sub["activity"], m1[sub.index])[0],
                spearmanr(sub["activity"], m2[sub.index])[0])
diffs = np.array([per_g[g][1] - per_g[g][0] for g in genes])
boot = [np.mean(rng.choice(diffs, len(diffs), replace=True)) for _ in range(10000)]
res["paired_bootstrap"] = {"mean_diff": float(np.mean(diffs)),
                           "p_leader_ge_challenger": float(np.mean([b >= 0 for b in boot])),
                           "n_genes": len(genes)}
json.dump(res, open("results/beat_azimuth_v7.json", "w"), indent=2)
print(json.dumps(res))
