"""Break attempt v2: Azimuth-style feature-rich GBT on the clean protocol.

Rule Set 2's advantage is its features: position-specific mono- and
di-nucleotides plus thermodynamics. We reproduce that feature family and
train HistGradientBoosting on RES-only, test V1/FC, mean per-gene Spearman.
"""
import json
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from crisprlib.featurize import gc_content, thermo_dg
from tools_transfer import load_human, load_mouse

DNA = "ACGT"
def azimuth_features(ctx: str) -> np.ndarray:
    feats = []
    # position-specific mononucleotides (30 x 4)
    for i, ch in enumerate(ctx):
        feats.extend(1.0 if ch == b else 0.0 for b in DNA)
    # position-specific dinucleotides (29 x 16)
    for i in range(len(ctx) - 1):
        di = ctx[i:i+2]
        feats.extend(1.0 if di == a + b else 0.0 for a in DNA for b in DNA)
    # global + thermodynamic
    feats.append(gc_content(ctx))
    feats.append(gc_content(ctx[4:24]))           # spacer GC
    feats.append(thermo_dg(ctx[4:24]))            # spacer duplex dG
    feats.append(thermo_dg(ctx[-9:]))             # PAM-proximal dG
    feats.append(ctx.count("GG") / 29.0)
    feats.append(ctx[24:27].count("G") / 3.0)     # PAM region G richness
    return np.array(feats, dtype=np.float32)

hu = load_human(); mo = load_mouse()
X_hu = np.stack([azimuth_features(s) for s in hu["context"]])
y_hu = hu["activity"].to_numpy(np.float32)
X_mo = np.stack([azimuth_features(s) for s in mo["context"]])
print("features", X_hu.shape)

gbt = HistGradientBoostingRegressor(max_iter=800, learning_rate=0.05,
                                    max_leaf_nodes=63, random_state=0).fit(X_hu, y_hu)
pred = gbt.predict(X_mo)
mo_r = mo.reset_index(drop=True)
per_gene = {}
for g, idx in mo_r.groupby("Gene Symbol").groups.items():
    idx = list(idx)
    r, _ = spearmanr(mo_r.loc[idx, "activity"], pred[mo_r.index.get_indexer(idx)])
    per_gene[g] = float(r)
mean_pg = float(np.mean(list(per_gene.values())))
out = {"model": "gbt_azimuth_features", "per_gene": per_gene, "mean_per_gene": mean_pg,
       "published_rule_set_2_fc": 0.52}
json.dump(out, open("results/beat_azimuth_v2.json", "w"), indent=2)
print(json.dumps(out, indent=2))
