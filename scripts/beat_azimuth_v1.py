"""Benchmark break attempt: Azimuth/Rule Set 2 cross-dataset protocol.

Azimuth (Doench 2016 / Fusi & Listgarten 2015) protocol: train on RES,
test on the FC dataset (= Doench V1), metric = mean per-gene Spearman.
Published Rule Set 2 result on FC: ~0.51-0.52 mean per-gene Spearman
(Azimuth bioRxiv 021568: "FC data (Spearman with/without new features
0.51 and 0.52)"). We run GuideCNN and SequenceGCN on the identical protocol
with the identical public datasets and report the same metric.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from crisprlib.featurize import one_hot, kmer_counts
from crisprlib.models import GuideCNN, SequenceGCN, kmer_graph
from crisprlib.train import predict, train_regressor
from sklearn.linear_model import Ridge
from tools_transfer import load_human, load_mouse

hu = load_human()   # RES-labelled rows (score_drug_gene_rank)
mo = load_mouse()   # V1 / FC dataset, Percent Rank labels
print(f"train RES {len(hu)}; test V1 {len(mo)} across {mo['Gene Symbol'].nunique()} genes")

def per_gene_spearman(df, yhat):
    vals = {}
    for g, idx in df.groupby("Gene Symbol").groups.items():
        idx = list(idx)
        if len(idx) < 20:
            continue
        r, _ = spearmanr(df.loc[idx, "activity"], yhat[df.index.get_indexer(idx)])
        vals[g] = float(r)
    return vals, float(np.mean(list(vals.values())))

X_hu = np.stack([one_hot(s, 30) for s in hu["context"]]).transpose(0, 2, 1)
y_hu = hu["activity"].to_numpy(np.float32)
X_mo = np.stack([one_hot(s, 30) for s in mo["context"]]).transpose(0, 2, 1)
K_hu = np.stack([kmer_counts(s, 3) for s in hu["context"]])
K_mo = np.stack([kmer_counts(s, 3) for s in mo["context"]])

# small internal val split for early stopping
rng = np.random.RandomState(0)
perm = rng.permutation(len(hu))
tr, val = perm[:int(0.9 * len(hu))], perm[int(0.9 * len(hu)):]

res = {"protocol": "Azimuth cross-dataset: train RES, test V1/FC, mean per-gene Spearman",
       "published_rule_set_2_fc": 0.52, "n_test_guides": len(mo)}

ridge = Ridge(alpha=1.0).fit(K_hu[tr], y_hu[tr])
pg, mean_pg = per_gene_spearman(mo.reset_index(drop=True), ridge.predict(K_mo))
res["ridge_kmer"] = {"per_gene": pg, "mean_per_gene": mean_pg}
print("ridge", mean_pg)

cnn = GuideCNN(seq_len=30, filters=32, kernel_widths=(3, 5, 7))
cnn, _ = train_regressor(cnn, X_hu[tr], y_hu[tr], X_hu[val], y_hu[val], epochs=60, lr=2e-3, seed=0)
pg, mean_pg = per_gene_spearman(mo.reset_index(drop=True), predict(cnn, X_mo))
res["cnn"] = {"per_gene": pg, "mean_per_gene": mean_pg}
print("cnn", mean_pg)

G_hu_tr = [kmer_graph(hu["context"].iloc[i], k=3) for i in tr]
G_hu_val = [kmer_graph(hu["context"].iloc[i], k=3) for i in val]
G_mo = [kmer_graph(s, k=3) for s in mo["context"]]
gnn = SequenceGCN(k=3, hidden=48, layers=2)
gnn, _ = train_regressor(gnn, G_hu_tr, y_hu[tr], G_hu_val, y_hu[val], epochs=60, lr=2e-3, seed=0, gnn=True)
pg, mean_pg = per_gene_spearman(mo.reset_index(drop=True), predict(gnn, G_mo, gnn=True))
res["gnn"] = {"per_gene": pg, "mean_per_gene": mean_pg}
print("gnn", mean_pg)

with open("results/beat_azimuth_v1.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps({k: v.get("mean_per_gene") for k, v in res.items() if isinstance(v, dict)}, indent=2))
