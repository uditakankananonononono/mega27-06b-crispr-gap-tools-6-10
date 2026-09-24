"""Second wave of published off-target scorer replications (IMPL) on crisprSQL,
same LOSO logistic harness as scripts/offtarget_scorers.py.

(c) CCTop-style (Stemmer 2015): mismatches weighted by distance from the PAM;
    we give the logistic model the PAM-distal distance of each mismatch and
    the count (CCTop's published premise: PAM-proximal mismatches disrupt
    cleavage more).
(d) CROP-IT-style (Singh 2015): exponentially decaying position weights;
    feature = sum over mismatches of exp(-dist_from_PAM/tau), tau=5, + count.
(e) CRISPRoff-style (Alkan 2018): guide:off-target hybrid binding free energy
    via ViennaRNA duplexfold (RNA-RNA approximation to the RNA-DNA hybrid, as
    in the original), + mismatch count.

Binary label = cleavage_freq > 0; LOSO over studies with >=10 pos and >=10 neg.
"""
import json
import numpy as np
import pandas as pd
import RNA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DNA = "ACGT"
COMP = str.maketrans("ACGT", "TGCA")

df = pd.read_csv("data/crisprsql/100720.csv")
df = df.dropna(subset=["cleavage_freq", "grna_target_sequence", "target_sequence"])
df["guide"] = df["grna_target_sequence"].str.upper().str[:20]
df["off"] = df["target_sequence"].str.upper().str.replace("N", "A").str[:20]
df = df[(df["guide"].str.len() == 20) & (df["off"].str.len() == 20)]
df = df[df["guide"].map(lambda s: set(s) <= set(DNA))]
df = df[df["off"].map(lambda s: set(s) <= set(DNA))]

def mismatches(g, o):
    return [i for i in range(20) if g[i] != o[i]]

pairs = df[["guide", "off", "study_name", "cleavage_freq"]].copy()
pairs["mm"] = [mismatches(g, o) for g, o in zip(pairs["guide"], pairs["off"])]
pairs = pairs[pairs["mm"].map(len) <= 6].reset_index(drop=True)
pairs["y"] = (pairs["cleavage_freq"] > 0).astype(int)
print(f"{len(pairs)} usable pairs", flush=True)

def feats_cctop(mm):
    v = np.zeros(3, dtype=np.float32)
    for i in mm:
        d = 20 - i  # distance from PAM (guide 3' end), 1..20
        v[0] += d
        v[1] = max(v[1], 21 - d)  # most PAM-proximal mismatch position
    v[2] = len(mm)
    return v

def feats_cropit(mm, tau=5.0):
    v = np.zeros(2, dtype=np.float32)
    for i in mm:
        v[0] += np.exp(-(20 - i) / tau)
    v[1] = len(mm)
    return v

def duplex_mfe(g, o):
    grna = g.replace("T", "U")
    target = o.translate(COMP)[::-1].replace("T", "U")
    return RNA.duplexfold(grna, target).energy

print("computing duplex energies ...", flush=True)
pairs["mfe"] = [duplex_mfe(g, o) for g, o in zip(pairs["guide"], pairs["off"])]
print("energies done", flush=True)

X = {
    "cctop_style": np.stack([feats_cctop(m) for m in pairs["mm"]]),
    "cropit_style": np.stack([feats_cropit(m) for m in pairs["mm"]]),
    "crisproff_style": np.stack([np.array([m, len(mm)], dtype=np.float32)
                                 for m, mm in zip(pairs["mfe"], pairs["mm"])]),
}
y = pairs["y"].to_numpy()
studies = pairs["study_name"].to_numpy()

res = {"n_pairs": int(len(pairs)), "studies": {}, "mean_auc": {},
       "_harness": "LOSO logistic, binary label cleavage_freq>0, same as offtarget_scorers.py",
       "_sources": {"cctop": "Stemmer 2015 PLoS ONE 10:e0144633",
                    "cropit": "Singh 2015 Genome Res 25:114",
                    "crisproff": "Alkan 2018 Nat Commun 9:4438"}}
accs = {k: [] for k in X}
for st in sorted(set(studies)):
    te = studies == st
    tr = ~te
    if y[te].sum() < 10 or (len(y[te]) - y[te].sum()) < 10:
        continue
    row = {"n": int(te.sum()), "n_pos": int(y[te].sum())}
    for name, F in X.items():
        m = LogisticRegression(max_iter=1000, C=1.0).fit(F[tr], y[tr])
        a = roc_auc_score(y[te], m.predict_proba(F[te])[:, 1])
        row[f"auc_{name}"] = round(float(a), 4)
        accs[name].append(a)
    res["studies"][st] = row
for name in X:
    res["mean_auc"][name] = round(float(np.mean(accs[name])), 4)
res["n_studies"] = len(res["studies"])
json.dump(res, open("results/offtarget_scorers2.json", "w"), indent=1)
print(json.dumps(res["mean_auc"], indent=1))
