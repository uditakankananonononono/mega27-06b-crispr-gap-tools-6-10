"""Break attempt v4: rank-averaged ensemble of Azimuth-style GBTs
(seq-only and seq+position features, 3 seeds each). Clean protocol."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.ensemble import HistGradientBoostingRegressor

from crisprlib.featurize import gc_content, thermo_dg

DNA = "ACGT"
def seq_features(ctx):
    feats = []
    for ch in ctx:
        feats.extend(1.0 if ch == b else 0.0 for b in DNA)
    for i in range(len(ctx) - 1):
        di = ctx[i:i+2]
        feats.extend(1.0 if di == a + b else 0.0 for a in DNA for b in DNA)
    feats += [gc_content(ctx), gc_content(ctx[4:24]), thermo_dg(ctx[4:24]),
              thermo_dg(ctx[-9:]), ctx.count("GG") / 29.0,
              ctx[24:27].count("G") / 3.0]
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

def build(df, pos):
    X = []
    for _, r in df.iterrows():
        f = seq_features(r["context"])
        if pos:
            f = f + [r["aa"] / 100.0, r["pct"] / 100.0]
        X.append(f)
    return np.array(X, dtype=np.float32)

y_hu = hu["score_drug_gene_rank"].to_numpy(np.float32)
preds = []
for pos in (False, True):
    X_hu = build(hu, pos); X_mo = build(mo, pos)
    for seed in (0, 1, 2):
        m = HistGradientBoostingRegressor(max_iter=1200, learning_rate=0.035,
                                          max_leaf_nodes=63, random_state=seed).fit(X_hu, y_hu)
        preds.append(rankdata(m.predict(X_mo)))
ens = np.mean(preds, axis=0)
mo = mo.reset_index(drop=True)
per_gene = {}
for g, idx in mo.groupby("gene").groups.items():
    idx = list(idx)
    r, _ = spearmanr(mo.loc[idx, "activity"], ens[mo.index.get_indexer(idx)])
    per_gene[g] = float(r)
mean_pg = float(np.mean(list(per_gene.values())))
out = {"model": "gbt_rank_ensemble_6x", "per_gene": per_gene,
       "mean_per_gene": mean_pg, "published_rule_set_2_fc": 0.52}
json.dump(out, open("results/beat_azimuth_v4.json", "w"), indent=2)
print(json.dumps(out, indent=2))
