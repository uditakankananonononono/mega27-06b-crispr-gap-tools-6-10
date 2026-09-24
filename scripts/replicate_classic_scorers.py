"""Replication implementations of three classic published CRISPR activity
scorers, trained and evaluated on the identical clean split as the Azimuth
ladder (train: RES human drug-treated 3,473; test: V1 mouse; mean per-gene
Spearman). These are METHOD-CLASS replications (IMPL), not official weights:
1. Rule Set 1 style (Doench 2014): logistic on position-specific mono- and
   dinucleotides over the 30-mer + GC count.
2. SSC style (Xu 2015): linear model on position-specific mononucleotides.
3. CRISPRscan style (Moreno-Mateos 2015): position-specific 6-mer linear
   model over the 20-nt guide (sparse one-hot, 15 x 4096).
Same loader/eval as scripts/beat_azimuth_v6.py.
"""
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge

DNA = "ACGT"

fc = pd.read_csv("data/FC_plus_RES_withPredictions.csv", sep=None, engine="python")
hu = fc[fc["drug"] != "nodrug"].copy()
hu["context"] = hu["30mer"].str.upper()
hu = hu[hu["context"].str.len() == 30].reset_index(drop=True)
y_hu = hu["score_drug_gene_rank"].to_numpy(np.float32)

v1 = pd.read_csv("data/V1_suppl_data.txt", sep="\t")
mo = pd.DataFrame({
    "context": v1["Extended Spacer(NNNN[20nt]NGGNNNNNNN)"].str.upper().str[:30],
    "activity": pd.to_numeric(v1["Percent Rank"], errors="coerce"),
    "gene": v1["Gene Symbol"],
}).dropna().reset_index(drop=True)
print(f"train {len(hu)}  test {len(mo)}", flush=True)

def per_gene_spearman(pred, df):
    vals = []
    for g, sub in df.groupby("gene"):
        if len(sub) < 5:
            continue
        r = spearmanr(sub["activity"], pred[sub.index])[0]
        if np.isfinite(r):
            vals.append(r)
    return float(np.mean(vals)), len(vals)

def mono_dinuc(ctxs, dinuc=True):
    rows = []
    for c in ctxs:
        f = [1.0 if ch == b else 0.0 for ch in c for b in DNA]
        if dinuc:
            f += [1.0 if c[i:i+2] == a + b else 0.0
                  for i in range(len(c) - 1) for a in DNA for b in DNA]
        g = c[4:24]
        f.append((g.count("G") + g.count("C")) / 20.0)
        rows.append(f)
    return np.array(rows, dtype=np.float32)

def sixmer_sparse(ctxs):
    n, r, cix = [], [], []
    for i, ctx in enumerate(ctxs):
        g = ctx[4:24]
        for p in range(15):
            k = g[p:p+6]
            if len(k) == 6 and set(k) <= set(DNA):
                v = 0
                for ch in k:
                    v = v * 4 + DNA.index(ch)
                n.append(i); r.append(1); cix.append(p * 4096 + v)
    return sparse.csr_matrix((np.ones(len(r)), (n, cix)),
                             shape=(len(ctxs), 15 * 4096), dtype=np.float32)

res = {}
# RS1-style logistic (mono+dinuc)
Xh = mono_dinuc(hu["context"].tolist()); Xm = mono_dinuc(mo["context"].tolist())
# RS1 was a logistic model on binarized activity (Doench 2014 Methods);
# binarize at the training median, score with P(active) for ranking.
y_bin = (y_hu > np.median(y_hu)).astype(int)
rs1 = LogisticRegression(max_iter=3000, C=1.0).fit(Xh, y_bin)
res["rs1_style_logistic"] = per_gene_spearman(rs1.predict_proba(Xm)[:, 1], mo)
print("rs1", res["rs1_style_logistic"], flush=True)
# SSC-style linear (mono only)
Xh = mono_dinuc(hu["context"].tolist(), dinuc=False)
Xm = mono_dinuc(mo["context"].tolist(), dinuc=False)
ssc = Ridge(alpha=1.0).fit(Xh, y_hu)
res["ssc_style_linear"] = per_gene_spearman(ssc.predict(Xm), mo)
print("ssc", res["ssc_style_linear"], flush=True)
# CRISPRscan-style 6-mer sparse linear
Xh = sixmer_sparse(hu["context"].tolist()); Xm = sixmer_sparse(mo["context"].tolist())
cs = Ridge(alpha=10.0).fit(Xh, y_hu)
res["crisprscan_style_6mer"] = per_gene_spearman(cs.predict(Xm), mo)
print("crisprscan", res["crisprscan_style_6mer"], flush=True)

res["_protocol"] = ("train RES human (drug!=nodrug), test V1 mouse, mean "
                    "per-gene Spearman, identical split as beat_azimuth_v6")
res["_reference_points"] = {"rs2_replication": 0.569, "our_v6_stack": 0.490}
json.dump(res, open("results/classic_scorers.json", "w"), indent=2)
print(json.dumps(res))
