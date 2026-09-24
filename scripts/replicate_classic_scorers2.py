"""Second wave of classic efficacy-scorer replications (IMPL), identical clean
split as scripts/replicate_classic_scorers.py (train RES human drug-treated,
test V1 mouse, mean per-gene Spearman).
4. Chari 2015 (sgRNA Designer 1.0) style: logistic on position-specific mono-
   and dinucleotide motifs over the 20-nt guide ONLY (no flanking context),
   activity binarized at the training median (Chari trained a classifier of
   high vs low activity).
5. WU-CRISPR (Wong 2015) style: structural + composition features - GC count,
   self-folding free energy (ViennaRNA fold), dinucleotide spectrum - logistic
   on the same binarized target.
"""
import json
import numpy as np
import pandas as pd
import RNA
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

DNA = "ACGT"

fc = pd.read_csv("data/FC_plus_RES_withPredictions.csv", sep=None, engine="python")
hu = fc[fc["drug"] != "nodrug"].copy()
hu["context"] = hu["30mer"].str.upper()
hu = hu[hu["context"].str.len() == 30].reset_index(drop=True)
hu["guide"] = hu["context"].str[4:24]
y_hu = hu["score_drug_gene_rank"].to_numpy(np.float32)
y_bin = (y_hu >= np.median(y_hu)).astype(int)

v1 = pd.read_csv("data/V1_suppl_data.txt", sep="\t")
mo = pd.DataFrame({
    "context": v1["Extended Spacer(NNNN[20nt]NGGNNNNNNN)"].str.upper().str[:30],
    "activity": pd.to_numeric(v1["Percent Rank"], errors="coerce"),
    "gene": v1["Gene Symbol"],
}).dropna().reset_index(drop=True)
mo["guide"] = mo["context"].str[4:24]
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

def chari_feats(guides):
    rows = []
    for g in guides:
        f = [1.0 if ch == b else 0.0 for ch in g for b in DNA]
        f += [1.0 if g[i:i+2] == a + b else 0.0
              for i in range(19) for a in DNA for b in DNA]
        f.append((g.count("G") + g.count("C")) / 20.0)
        rows.append(f)
    return np.array(rows, dtype=np.float32)

def wu_feats(guides):
    rows = []
    for g in guides:
        mfe = RNA.fold_compound(g.replace("T", "U")).mfe()[1]
        gc = (g.count("G") + g.count("C")) / 20.0
        di = [g.count(a + b) / 19.0 for a in DNA for b in DNA]
        gg = 1.0 if g[19] == "G" else 0.0  # position-20 G (Wong: PAM-adjacent G)
        rows.append([gc, mfe, gg] + di)
    return np.array(rows, dtype=np.float32)

print("featurizing ...", flush=True)
Xc_tr, Xc_te = chari_feats(hu["guide"]), chari_feats(mo["guide"])
Xw_tr, Xw_te = wu_feats(hu["guide"]), wu_feats(mo["guide"])

out = {}
for name, Xtr, Xte in [("chari_style_logistic", Xc_tr, Xc_te),
                       ("wu_crispr_style", Xw_tr, Xw_te)]:
    m = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y_bin)
    p = m.predict_proba(Xte)[:, 1]
    rho, ng = per_gene_spearman(pd.Series(p, index=mo.index), mo)
    out[name] = [round(rho, 4), ng]
    print(name, out[name], flush=True)

out["_protocol"] = ("identical to scripts/replicate_classic_scorers.py: "
                    "train RES human (drug != nodrug), test V1 mouse, mean per-gene Spearman; "
                    "IMPL method-class replications, not official weights")
out["_sources"] = {"chari": "Chari 2015 Nat Methods 12:823",
                   "wu_crispr": "Wong 2015 Genome Biol 16:218"}
json.dump(out, open("results/classic_scorers2.json", "w"), indent=1)
print("done")
