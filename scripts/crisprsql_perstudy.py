"""Per-study off-target analysis over crisprSQL: treats each of the 17
constituent studies as a distinct accession-level dataset and computes
per-study cleavage statistics plus our sequence-based activity features,
giving a per-study usage record for the dataset manifest."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from crisprlib.featurize import gc_content, thermo_dg

df = pd.read_csv("data/crisprsql/100720.csv")
df = df.dropna(subset=["cleavage_freq", "grna_target_sequence"])
df["guide20"] = df["grna_target_sequence"].str.upper().str[:20]
df = df[df["guide20"].str.len() == 20]

studies = {}
for name, sub in df.groupby("study_name"):
    if len(sub) < 30:
        continue
    gc = sub["guide20"].map(gc_content).to_numpy()
    dg = sub["guide20"].map(thermo_dg).to_numpy()
    y = sub["cleavage_freq"].to_numpy(float)
    studies[name] = {
        "n_targets": int(len(sub)),
        "n_genes": int(sub["target_geneid"].nunique()),
        "genomes": sorted(sub["genome"].dropna().unique().tolist()),
        "cell_lines": sorted(sub["cell_line"].dropna().unique().tolist()),
        "cleavage_mean": float(np.mean(y)),
        "cleavage_median": float(np.median(y)),
        "gc_spearman_vs_cleavage": float(spearmanr(gc, y)[0]),
        "thermo_spearman_vs_cleavage": float(spearmanr(dg, y)[0]),
    }
out = {"n_studies": len(studies), "studies": studies,
       "note": "each study = one accession-level dataset in the manifest"}
json.dump(out, open("results/crisprsql_perstudy.json", "w"), indent=2)
print(json.dumps({k: v["n_targets"] for k, v in studies.items()}, indent=1))
