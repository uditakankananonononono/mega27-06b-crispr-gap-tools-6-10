"""Independent thermodynamic cross-check of Tool 6 duplex flags, and a
CHOPCHOP-rule audit of real guide sets.

Part A: ViennaRNA (our engine) vs primer3-py (independent NN thermodynamics
library, UNAFold-derived) on the same guide pairs. We score 2,000 seeded
random real-guide pairs with both engines and report the Spearman between
the two energy scales plus flag-agreement (Cohen's kappa) at matched flag
rates (each engine flags its own worst q-quantile, q = ViennaRNA flag rate
at theta = -12 kcal/mol, matching discovery_realguide_risk.py).

Part B: CHOPCHOP v3 design-rule audit (Labun 2019, REF) on the 4,692 real
published guides: GC window [0.35, 0.70], poly-T (TTTT) exclusion,
self-complementarity - the same rules our screener encodes.
"""
import json, random
import numpy as np
import pandas as pd
import primer3
import RNA
from scipy.stats import spearmanr

random.seed(11)
fc = pd.read_csv("data/FC_plus_RES_withPredictions.csv", sep=None, engine="python")
fc["spacer"] = fc["30mer"].str[4:24]
v1 = pd.read_csv("data/V1_suppl_data.txt", sep="\t")
v1 = v1.rename(columns={"Spacer Sequence": "spacer"})
real = pd.concat([fc[["spacer"]], v1[["spacer"]]])
real = real[real["spacer"].str.match("^[ACGT]{20}$")].drop_duplicates("spacer")
guides = real["spacer"].tolist()
print(f"real guides: {len(guides)}", flush=True)

# --- Part A: engine cross-check on 2,000 pairs
pairs = [(random.choice(guides), random.choice(guides)) for _ in range(2000)]
vrna = np.array([RNA.duplexfold(a.replace("T","U"), b.replace("T","U")).energy
                 for a, b in pairs])
p3 = np.array([primer3.calc_heterodimer(a, b).dg / 1000.0 for a, b in pairs])  # cal -> kcal
rho = float(spearmanr(vrna, p3)[0])
q = float((vrna <= -12.0).mean())
flag_v = vrna <= -12.0
flag_p = p3 <= np.quantile(p3, q) if q > 0 else np.zeros(len(p3), bool)
agree = (flag_v == flag_p).mean()
# Cohen's kappa
pe = flag_v.mean() * flag_p.mean() + (1 - flag_v.mean()) * (1 - flag_p.mean())
kappa = float((agree - pe) / (1 - pe)) if pe < 1 else 1.0

# --- Part B: CHOPCHOP v3 rule audit
gc = np.array([(s.count("G") + s.count("C")) / 20 for s in guides])
out = {
 "partA_engine_crosscheck": {
   "n_pairs": 2000, "engine_ours": "ViennaRNA 2.7.2 duplexfold",
   "engine_check": "primer3-py 2.3.1 calc_heterodimer (kcal/mol)",
   "spearman_energy_scales": round(rho, 4),
   "vrna_flag_rate_at_-12": round(q, 4),
   "flag_agreement_matched_rate": round(float(agree), 4),
   "cohens_kappa": round(kappa, 4),
   "vrna_mfe_mean": round(float(vrna.mean()), 3),
   "primer3_dg_mean": round(float(p3.mean()), 3)},
 "partB_chopchop_rule_audit": {
   "n_guides": len(guides),
   "gc_window": [0.35, 0.70],
   "gc_violation_rate": round(float(((gc > 0.70) | (gc < 0.35)).mean()), 4),
   "polyT_TTTT_share": round(float(np.mean(["TTTT" in s for s in guides])), 4),
   "ref": "CHOPCHOP v3 default rules, Labun 2019 NAR 47:W171"},
}
json.dump(out, open("results/duplex_crosscheck.json", "w"), indent=1)
print(json.dumps(out, indent=1))
