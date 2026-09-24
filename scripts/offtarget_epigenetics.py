"""Does crisprSQL's epigenetic track data improve off-target prediction?
LOSO logistic: CFD-style sequence features vs CFD-style + 5 epigenetic
features (CTCF, DNase, RRBS methylation, H3K4me3, DRIP from crisprSQL).
Same pairs/labels/protocol as scripts/offtarget_scorers.py.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DNA = "ACGT"
EPI = ["epigen_ctcf", "epigen_dnase", "epigen_rrbs", "epigen_h3k4me3", "epigen_drip"]
df = pd.read_csv("data/crisprsql/100720.csv")
df = df.dropna(subset=["cleavage_freq", "grna_target_sequence", "target_sequence"] + EPI)
df["guide"] = df["grna_target_sequence"].str.upper().str[:20]
df["off"] = df["target_sequence"].str.upper().str.replace("N", "A").str[:20]
df = df[(df["guide"].str.len() == 20) & (df["off"].str.len() == 20)]
df = df[df["guide"].map(lambda s: set(s) <= set(DNA))]
df = df[df["off"].map(lambda s: set(s) <= set(DNA))]
df["mm"] = [[(i, g[i], o[i]) for i in range(20) if g[i] != o[i]]
            for g, o in zip(df["guide"], df["off"])]
df = df[df["mm"].map(len) <= 6].reset_index(drop=True)
df["y"] = (df["cleavage_freq"] > 0).astype(int)
print(f"{len(df)} pairs with complete epigenetics", flush=True)

subs = [(a, b) for a in DNA for b in DNA if a != b]
def feats_cfd(mm):
    v = np.zeros(20 * 12, dtype=np.float32)
    for i, a, b in mm:
        v[i * 12 + subs.index((a, b))] = 1.0
    return np.append(v, len(mm) / 6.0)

X_seq = np.stack([feats_cfd(m) for m in df["mm"]])
X_epi = df[EPI].to_numpy(np.float32)
X_epi = np.nan_to_num(np.log1p(np.clip(X_epi, 0, None)))
X_full = np.hstack([X_seq, X_epi])
y = df["y"].to_numpy(); studies = df["study_name"].to_numpy()

res = {"n_pairs": int(len(df)), "studies": {}, "mean_auc": {}}
a_s, a_f = [], []
for st in sorted(set(studies)):
    te = studies == st; tr = ~te
    if y[te].sum() < 10 or (te.sum() - y[te].sum()) < 10 or y[tr].sum() < 50:
        continue
    s = roc_auc_score(y[te], LogisticRegression(max_iter=1000, C=0.5)
                      .fit(X_seq[tr], y[tr]).predict_proba(X_seq[te])[:, 1])
    f = roc_auc_score(y[te], LogisticRegression(max_iter=1000, C=0.5)
                      .fit(X_full[tr], y[tr]).predict_proba(X_full[te])[:, 1])
    res["studies"][st] = {"n": int(te.sum()), "auc_seq_only": float(s),
                          "auc_seq_plus_epigenetics": float(f)}
    a_s.append(s); a_f.append(f)
    print(st, round(s, 3), round(f, 3), flush=True)
res["mean_auc"] = {"seq_only": float(np.mean(a_s)),
                   "seq_plus_epigenetics": float(np.mean(a_f)),
                   "n_studies": len(a_s)}
json.dump(res, open("results/offtarget_epigenetics.json", "w"), indent=2)
print(json.dumps(res["mean_auc"]))
