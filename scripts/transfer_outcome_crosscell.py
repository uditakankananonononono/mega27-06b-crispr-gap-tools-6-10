"""Cross-cell-line / cross-species transfer of Tool 7 repair-outcome models.

Train: inDelphi U2OS Lib-A (all sites passing read filters).
Test:  (a) U2OS Lib-B (same cell line, disjoint loci),
       (b) mESC Lib-B (different cell line AND species).
Also runs Lindel zero-shot on the same test sites for reference.

Datasets: figshare 6837956 files 13502321 (U2OS LibA rep1), 13502330
(U2OS LibB rep1), 13769981 (mESC LibB Cas9 r1); contexts/guides from
maxwshen/inDelphi-dataprocessinganalysis data-libprocessing.
"""
import json
import pickle
import re
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "data/lindel")
sys.path.insert(0, ".")
from Predictor import gen_prediction  # noqa: E402
from tools_outcome.loader import aggregate_events, load_targets  # noqa: E402
from crisprlib.featurize import gc_content, microhomology_score, one_hot, thermo_dg  # noqa: E402

wb = pickle.load(open("data/lindel/Model_weights.pkl", "rb"), encoding="latin1")
prereq = pickle.load(open("data/lindel/model_prereq.pkl", "rb"), encoding="latin1")
label, frame_shift = prereq[0], prereq[3]
MASK_RE = re.compile(r"^-(28|29|30)\+")
INS1_RE = re.compile(r"^1\+[ACGT]$")

def lindel_predict(ctx, guide):
    i = ctx.find(guide)
    if i != 10 or len(ctx) < 55:
        return None
    up = i + 17
    seq = "A" * (30 - up) + ctx
    seq = seq + "A" * (65 - len(seq))
    out = gen_prediction(seq, wb, prereq)
    if isinstance(out, str):
        return None
    v = out[0].copy()
    for cls, idx in label.items():
        if MASK_RE.match(cls):
            v[idx] = 0.0
    tot = v.sum()
    if tot <= 0:
        return None
    v /= tot
    return (sum(v[idx] for cls, idx in label.items() if INS1_RE.match(cls)),
            float(np.dot(v, frame_shift)))

def positional_features(ctx: str) -> np.ndarray:
    oh = one_hot(ctx, 55).reshape(-1)
    left, right = ctx[:27], ctx[27:]          # empirical cut (grna alignment)
    feats = [microhomology_score(left, right), gc_content(ctx),
             gc_content(left[-10:]), thermo_dg(left[-15:]),
             thermo_dg(right[:15])]
    for d in (2, 3, 4, 5, 7, 10, 15):
        feats.append(microhomology_score(left[:-d], right[d:])
                     if len(left) > d and len(right) > d else 0.0)
    return np.concatenate([oh, np.array(feats, dtype=np.float32)])

def load_lib(tpath, gpath, csv):
    tg = load_targets(tpath)
    gr = [l.strip().upper() for l in open(gpath) if l.strip()]
    sites = aggregate_events(csv, tg)
    ctx2exp = {s: i for i, s in enumerate(tg)}
    exp = [ctx2exp[s.context] for s in sites]
    return sites, gr, exp

tr_sites, _, _ = load_lib("data/targets-libA.txt", "data/grna-libA.txt",
                          "data/U2OS_LibA_postCas9_rep1.csv")
Xtr = np.stack([positional_features(s.context) for s in tr_sites])
print(f"train: {len(tr_sites)} U2OS Lib-A sites")

res = {"train": {"set": "U2OS Lib-A rep1", "n": len(tr_sites)}, "test": {}}
for name, tpath, gpath, csv in [
    ("U2OS_Lib-B", "data/targets-libB.txt", "data/grna-libB.txt",
     "data/U2OS_LibB_postCas9_rep1.csv"),
    ("mESC_Lib-B", "data/targets-libB.txt", "data/grna-libB.txt",
     "data/mESC_LibB_Cas9_r1.csv"),
]:
    sites, gr, exp = load_lib(tpath, gpath, csv)
    X = np.stack([positional_features(s.context) for s in sites])
    entry = {"n_sites": len(sites)}
    for tname in ("ins1_frac", "frameshift_frac"):
        ytr = np.array([getattr(s, tname) for s in tr_sites], dtype=np.float32)
        gbt = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05,
                                            random_state=0).fit(Xtr, ytr)
        ours = gbt.predict(X)
        y = np.array([getattr(s, tname) for s in sites])
        obs_l, our_l, lin_l = [], [], []
        for j, s in enumerate(sites):
            p = lindel_predict(s.context, gr[exp[j]])
            if p is None:
                continue
            pred = p[0] if tname == "ins1_frac" else p[1]
            obs_l.append(float(y[j])); our_l.append(float(ours[j]))
            lin_l.append(float(pred))
        entry[tname] = {
            "n_compared": len(obs_l),
            "ours_gbt_spearman": float(spearmanr(obs_l, our_l)[0]),
            "lindel_zeroshot_spearman": float(spearmanr(obs_l, lin_l)[0]),
        }
        print(name, tname, json.dumps(entry[tname]))
    res["test"][name] = entry

json.dump(res, open("results/outcome_crosscell.json", "w"), indent=2)
print("wrote results/outcome_crosscell.json")
