"""Discovery attempt: design rules for multiplex-safe guide sets.

Physics-based analysis over 600 random spacers: which sequence properties
predict (a) strong self-folding, (b) cross-guide duplex flags? Rules are
extracted from ViennaRNA ground truth, not fitted to labels. Live (not CI).
"""
import json

import numpy as np
from scipy.stats import pointbiserialr, spearmanr

from crisprlib.featurize import gc_content
from tools_multiplex.duplex import heteroduplex_mfe, self_fold_mfe

rng = np.random.default_rng(11)
N = 600
seqs = []
while len(seqs) < N:
    gc = rng.uniform(0.25, 0.75)
    p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
    seqs.append("".join(rng.choice(list("ACGT"), size=20, p=p)))

gcs = np.array([gc_content(s) for s in seqs])
sf = np.array([self_fold_mfe(s) for s in seqs])
homo = np.array([max(len(run) for base in "ACGT"
                     for run in s.split(base) if False) if False else
                 max((len(list(g)) for _, g in __import__("itertools").groupby(s)), default=0)
                 for s in seqs])

# (a) self-fold drivers
res = {"self_fold": {
    "spearman_gc": float(spearmanr(gcs, sf).statistic),
    "spearman_homopolymer": float(spearmanr(homo, sf).statistic),
    "mean_mfe": float(sf.mean()),
    "frac_strong_self_fold": float((sf <= -6.0).mean()),
}}

# (b) pairwise duplex flags on a subsample
sub = seqs[:120]
flags, gc_pairs = [], []
for i in range(len(sub)):
    for j in range(i + 1, len(sub)):
        e = heteroduplex_mfe(sub[i], sub[j])
        flags.append(1.0 if e <= -12.0 else 0.0)
        gc_pairs.append(abs(gcs[i] - gcs[j]))
flags = np.array(flags)
res["duplex"] = {
    "n_pairs": int(len(flags)),
    "frac_flagged": float(flags.mean()),
    "pointbiserial_gc_delta": float(pointbiserialr(flags, gc_pairs).statistic),
}

# rule candidates
low_gc = gcs < 0.35
high_gc = gcs > 0.70
res["rules"] = {
    "low_GC_self_fold_rate": float((sf[low_gc] <= -6.0).mean()) if low_gc.any() else None,
    "high_GC_self_fold_rate": float((sf[high_gc] <= -6.0).mean()) if high_gc.any() else None,
    "mid_GC_self_fold_rate": float((sf[~(low_gc | high_gc)] <= -6.0).mean()),
    "homopolymer4plus_self_fold_rate": float((sf[homo >= 4] <= -6.0).mean()),
}
print(json.dumps(res, indent=2))
json.dump(res, open("results/multiplex_rules.json", "w"), indent=2)
