"""Off-target scorer replications (method classes, IMPL) on crisprSQL.

Models, trained as logistic regression with leave-one-study-out (LOSO)
evaluation, binary label = cleavage_freq > 0:
(a) Hsu/MIT style: mismatch-position features only (20 dims).
(b) CFD style (Doench 2016): mismatch position x base-identity features
    (20 x 12 one-hot) + mismatch count.
Baselines: GC content and guide thermo_dg (already in per-study results).
Metric: ROC AUC per held-out study (studies with >= 10 positives and >= 10
negatives), mean across studies. Real data: data/crisprsql/100720.csv
(25,632 guide/off-target pairs across 17 studies).
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DNA = "ACGT"
df = pd.read_csv("data/crisprsql/100720.csv")
df = df.dropna(subset=["cleavage_freq", "grna_target_sequence", "target_sequence"])
df["guide"] = df["grna_target_sequence"].str.upper().str[:20]
df["off"] = df["target_sequence"].str.upper().str.replace("N", "A").str[:20]
df = df[(df["guide"].str.len() == 20) & (df["off"].str.len() == 20)]
df = df[df["guide"].map(lambda s: set(s) <= set(DNA))]
df = df[df["off"].map(lambda s: set(s) <= set(DNA))]

def mismatches(g, o):
    return [(i, g[i], o[i]) for i in range(20) if g[i] != o[i]]

pairs = df[["guide", "off", "study_name", "cleavage_freq"]].copy()
pairs["mm"] = [mismatches(g, o) for g, o in zip(pairs["guide"], pairs["off"])]
pairs = pairs[pairs["mm"].map(len) <= 6].reset_index(drop=True)
pairs["y"] = (pairs["cleavage_freq"] > 0).astype(int)
print(f"{len(pairs)} usable pairs", flush=True)

def feats_hsu(mm):
    v = np.zeros(20, dtype=np.float32)
    for i, _, _ in mm:
        v[i] = 1.0
    return v

def feats_cfd(mm):
    v = np.zeros(20 * 12, dtype=np.float32)
    subs = [(a, b) for a in DNA for b in DNA if a != b]
    for i, a, b in mm:
        v[i * 12 + subs.index((a, b))] = 1.0
    return np.append(v, len(mm) / 6.0)

X_hsu = np.stack([feats_hsu(m) for m in pairs["mm"]])
X_cfd = np.stack([feats_cfd(m) for m in pairs["mm"]])
y = pairs["y"].to_numpy()
studies = pairs["study_name"].to_numpy()

res = {"n_pairs": int(len(pairs)), "studies": {}, "mean_auc": {}}
auc_hsu, auc_cfd = [], []
for st in sorted(set(studies)):
    te = studies == st
    tr = ~te
    if y[te].sum() < 10 or (len(y[te]) - y[te].sum()) < 10:
        continue
    if y[tr].sum() < 50:
        continue
    a_hsu = roc_auc_score(y[te], LogisticRegression(max_iter=1000, C=0.5)
                          .fit(X_hsu[tr], y[tr]).predict_proba(X_hsu[te])[:, 1])
    a_cfd = roc_auc_score(y[te], LogisticRegression(max_iter=1000, C=0.5)
                          .fit(X_cfd[tr], y[tr]).predict_proba(X_cfd[te])[:, 1])
    res["studies"][st] = {"n": int(te.sum()), "n_pos": int(y[te].sum()),
                          "auc_hsu_style": float(a_hsu),
                          "auc_cfd_style": float(a_cfd)}
    auc_hsu.append(a_hsu); auc_cfd.append(a_cfd)
    print(st, round(a_hsu, 3), round(a_cfd, 3), flush=True)
res["mean_auc"] = {"hsu_style": float(np.mean(auc_hsu)),
                   "cfd_style": float(np.mean(auc_cfd)),
                   "n_studies": len(auc_hsu)}
json.dump(res, open("results/offtarget_scorers.json", "w"), indent=2)
print(json.dumps(res["mean_auc"]))
