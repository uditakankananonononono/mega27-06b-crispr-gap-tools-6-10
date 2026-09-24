"""Discovery: heteroduplex and self-fold risk among REAL published guide sets.

Claim under test: same-gene guide pairs (the natural multiplex sets people
co-deliver) carry quantifiable heterodimer risk, distinct from random-spacer
baseline (10.8% flagged at <= -12 kcal/mol) and from cross-gene real-guide
controls. Also audits GC-rule violation and self-fold rates in real screens.
"""
import json, random, sys
import pandas as pd
import RNA

random.seed(7)
DUPLEX_CUT = -12.0   # kcal/mol strong heteroduplex
SELF_CUT = -5.0      # kcal/mol strong self-fold

def revcomp(s):
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]

def dup_mfe(a, b):
    return RNA.duplexfold(a, revcomp(b)).energy

def self_mfe(a):
    return RNA.duplexfold(a, revcomp(a)).energy

fc = pd.read_csv("data/FC_plus_RES_withPredictions.csv", sep=None, engine="python")
fc["spacer"] = fc["30mer"].str[4:24]
v1 = pd.read_csv("data/V1_suppl_data.txt", sep="\t")
v1 = v1.rename(columns={"Spacer Sequence": "spacer", "Gene Symbol": "Target gene"})
real = pd.concat([fc[["spacer", "Target gene"]].assign(species="human"),
                  v1[["spacer", "Target gene"]].assign(species="mouse")])
real = real[real["spacer"].str.match("^[ACGT]{20}$")].drop_duplicates("spacer")
print(f"real guides: {len(real)}", file=sys.stderr)

# --- self-fold audit over all real guides
sf = [self_mfe(s) for s in real["spacer"]]
real["self_mfe"] = sf
gc = real["spacer"].apply(lambda s: (s.count("G") + s.count("C")) / 20)
out = {
  "n_real_guides": int(len(real)),
  "real_strong_self_fold_rate": float((real["self_mfe"] <= SELF_CUT).mean()),
  "real_mean_self_mfe": float(real["self_mfe"].mean()),
  "gc_rule_violation_rate": float(((gc > 0.70) | (gc < 0.35)).mean()),
  "high_gc_share": float((gc > 0.70).mean()),
  "low_gc_share": float((gc < 0.35).mean()),
}

# --- same-gene pairs (subsample <=100 guides/gene) vs cross-gene controls
rng = random.Random(7)
same_pairs, cross_pairs = [], []
for (gene, species), grp in real.groupby(["Target gene", "species"]):
    guides = grp["spacer"].tolist()
    if len(guides) > 100:
        guides = rng.sample(guides, 100)
    for i in range(len(guides)):
        for j in range(i + 1, len(guides)):
            same_pairs.append((guides[i], guides[j]))
allg = real["spacer"].tolist()
rng.shuffle(allg)
target_n = min(len(same_pairs), 30000)
same_pairs = rng.sample(same_pairs, min(len(same_pairs), 30000))
seen = set()
while len(cross_pairs) < target_n:
    a, b = rng.sample(allg, 2)
    ga = real.loc[real["spacer"] == a, "Target gene"].iloc[0]
    gb = real.loc[real["spacer"] == b, "Target gene"].iloc[0]
    if ga != gb and (a, b) not in seen:
        seen.add((a, b)); cross_pairs.append((a, b))

def flag_rate(pairs):
    n_flag = 0
    for k, (a, b) in enumerate(pairs):
        if dup_mfe(a, b) <= DUPLEX_CUT:
            n_flag += 1
        if k % 5000 == 0:
            print(f"  {k}/{len(pairs)}", file=sys.stderr)
    return n_flag / max(1, len(pairs))

print(f"same-gene pairs: {len(same_pairs)}", file=sys.stderr)
out["same_gene_pairs"] = len(same_pairs)
out["same_gene_flag_rate"] = flag_rate(same_pairs)
print(f"cross-gene pairs: {len(cross_pairs)}", file=sys.stderr)
out["cross_gene_pairs"] = len(cross_pairs)
out["cross_gene_flag_rate"] = flag_rate(cross_pairs)
out["random_spacer_baseline"] = 0.10812324929971989

with open("results/realguide_risk.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
