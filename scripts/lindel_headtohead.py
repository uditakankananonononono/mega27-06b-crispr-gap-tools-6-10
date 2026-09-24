"""Head-to-head: official Lindel model (Chen et al. 2019, NAR 47:7989) vs our
Tool 7 GBT positional model, both predicting inDelphi U2OS Lib-A held-out
targets (Shen et al. 2018, Nature 563:646).

Lindel is run ZERO-SHOT from its published weights (data/lindel/, downloaded
from github.com/shendurelab/Lindel). Our GBT is trained on the same 80/20
split (seed 0) as scripts/train_outcome_model.py.

Geometry (empirically calibrated): Lib-A guides align at ctx[10:30] in
1996/2000 targets (data/grna-libA.txt), so the PAM is ctx[30:33] and the
Cas9 cut is between indices 26/27. Lindel expects a 65-mer with guide at
[13:33] and cut between 29/30, so we pad 3 nt left / 6 nt right. Predicted
deletion classes extending past the real context ('-28+*', '-29+*', '-30+*')
are masked and the distribution renormalized; masked mass is reported.
"""
import json
import pickle
import re
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

sys.path.insert(0, "data/lindel")
sys.path.insert(0, ".")
from Predictor import gen_prediction  # noqa: E402
from tools_outcome.loader import aggregate_events, load_targets  # noqa: E402
from crisprlib.featurize import gc_content, microhomology_score, one_hot, thermo_dg  # noqa: E402

wb = pickle.load(open("data/lindel/Model_weights.pkl", "rb"), encoding="latin1")
prereq = pickle.load(open("data/lindel/model_prereq.pkl", "rb"), encoding="latin1")
label = prereq[0]          # class string -> index
frame_shift = prereq[3]    # frameshift indicator per class

tg = load_targets("data/targets-libA.txt")
grna = [l.strip().upper() for l in open("data/grna-libA.txt") if l.strip()]
sites = aggregate_events("data/U2OS_LibA_postCas9_rep1.csv", tg)
# aggregate_events keeps only targets passing read-depth filters; recover the
# target index per site by matching context (contexts are unique in Lib-A).
ctx2exp = {s: i for i, s in enumerate(tg)}
site_exp = [ctx2exp[s.context] for s in sites]

MASK_RE = re.compile(r"^-(28|29|30)\+")   # pad-touching deletion classes
INS1_RE = re.compile(r"^1\+[ACGT]$")

def lindel_predict(ctx, guide):
    """Zero-shot Lindel prediction on a 56-nt Lib-A context, or None."""
    i = ctx.find(guide)
    if i != 10 or len(ctx) < 55:
        return None
    up = i + 17                       # cut between up-1 and up (canonical, 3nt 5' of PAM)
    leftpad = 30 - up
    if leftpad < 0:
        return None
    seq = "A" * leftpad + ctx
    seq = seq + "A" * (65 - len(seq))
    assert seq[13:33] == guide and seq[33:36] == ctx[30:33]
    out = gen_prediction(seq, wb, prereq)
    if isinstance(out, str):
        return None
    y_hat, _ = out
    v = y_hat.copy()
    masked = 0.0
    for cls, idx in label.items():
        if MASK_RE.match(cls):
            masked += v[idx]
            v[idx] = 0.0
    tot = v.sum()
    if tot <= 0:
        return None
    v = v / tot
    ins1 = sum(v[idx] for cls, idx in label.items() if INS1_RE.match(cls))
    fs = float(np.dot(v, frame_shift))
    return ins1, fs, float(masked)

def positional_features(ctx: str) -> np.ndarray:
    oh = one_hot(ctx, 55).reshape(-1)
    left, right = ctx[:28], ctx[28:]
    feats = [microhomology_score(left, right), gc_content(ctx),
             gc_content(left[-10:]), thermo_dg(left[-15:]),
             thermo_dg(right[:15])]
    for d in (2, 3, 4, 5, 7, 10, 15):
        if len(left) > d and len(right) > d:
            feats.append(microhomology_score(left[:-d], right[d:]))
        else:
            feats.append(0.0)
    return np.concatenate([oh, np.array(feats, dtype=np.float32)])

X = np.stack([positional_features(s.context) for s in sites])
idx = np.arange(len(sites))
tr, te = train_test_split(idx, test_size=0.2, random_state=0)

res = {"n_sites": len(sites), "n_test": len(te), "targets": {}}
for tname in ("ins1_frac", "frameshift_frac"):
    y = np.array([getattr(s, tname) for s in sites], dtype=np.float32)
    gbt = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05,
                                        random_state=0).fit(X[tr], y[tr])
    ours = gbt.predict(X[te])
    # Lindel zero-shot on the same test targets
    obs_l, our_l, lin_l, mask_fracs = [], [], [], []
    for j, i in enumerate(te):
        s = sites[i]
        p = lindel_predict(s.context, grna[site_exp[i]])
        if p is None:
            continue
        ins1, fs, masked = p
        pred = ins1 if tname == "ins1_frac" else fs
        obs_l.append(float(y[i])); our_l.append(float(ours[j]))
        lin_l.append(float(pred)); mask_fracs.append(masked)
    res["targets"][tname] = {
        "n_compared": len(obs_l),
        "lindel_zeroshot_spearman": float(spearmanr(obs_l, lin_l)[0]),
        "ours_gbt_spearman": float(spearmanr(obs_l, our_l)[0]),
        "mean_masked_mass_lindel": float(np.mean(mask_fracs)),
    }
    print(tname, json.dumps(res["targets"][tname]))

res["geometry"] = {
    "guide_alignment": "ctx[10:30] via grna-libA.txt exact match (1996/2000)",
    "cut": "between ctx indices 26/27 (canonical, 3 nt 5' of ctx[30:33] PAM)",
    "lindel_input": "3 nt left pad + right pad to 65-mer; pad-touching "
                    "deletion classes masked and renormalized",
}
res["provenance"] = {
    "lindel_weights": "github.com/shendurelab/Lindel Model_weights.pkl + "
                      "model_prereq.pkl (Chen 2019 NAR 47:7989)",
    "observed": "figshare 6837956 U2OS_+_LibA_postCas9_rep1.csv (Shen 2018)",
}
json.dump(res, open("results/lindel_headtohead.json", "w"), indent=2)
print("wrote results/lindel_headtohead.json")
