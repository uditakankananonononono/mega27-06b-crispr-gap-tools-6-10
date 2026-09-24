"""Generate paper figures from results JSONs (live, not CI)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Fig 1: benchmark bars across tools
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
d = json.load(open("results/duplex_surrogate.json"))
ax = axes[0]
models = ["ridge_kmer", "cnn", "gnn"]
ax.bar(range(3), [d[m]["spearman"] for m in models], color=["#888", "#3366aa", "#aa6633"])
ax.set_xticks(range(3)); ax.set_xticklabels(["Ridge", "CNN", "GCN"], fontsize=8)
ax.set_title("Tool 6: duplex-MFE surrogate\n(vs ViennaRNA teacher)", fontsize=9)
ax.set_ylabel("Held-out Spearman"); ax.set_ylim(0, 0.8)

o = json.load(open("results/outcome_models.json"))
ax = axes[1]
targets = ["frameshift_frac", "ins1_frac", "mh_del_frac"]
x = np.arange(3); w = 0.25
for i, m in enumerate(["gbt_positional", "cnn", "gnn"]):
    ax.bar(x + i * w, [o["targets"][t][m]["spearman"] for t in targets], w,
           label={"gbt_positional": "GBT", "cnn": "CNN", "gnn": "GCN"}[m])
ax.set_xticks(x + w); ax.set_xticklabels(["frameshift", "+1 insertion", "MH deletion"], fontsize=8)
ax.set_title("Tool 7: repair-outcome prediction\n(inDelphi U2OS, 349 held-out sites)", fontsize=9)
ax.legend(fontsize=7); ax.set_ylim(-0.1, 0.7)

t = json.load(open("results/transfer.json"))
ax = axes[2]
x = np.arange(3); w = 0.35
models = ["ridge_kmer", "cnn", "gnn"]
ax.bar(x - w/2, [t[m]["human_heldout"]["spearman"] for m in models], w, label="human held-out")
ax.bar(x + w/2, [t[m]["mouse_transfer"]["spearman"] for m in models], w, label="mouse transfer")
ax.set_xticks(x); ax.set_xticklabels(["Ridge", "CNN", "GCN"], fontsize=8)
ax.set_title("Tool 10: human->mouse transfer\n(Doench FC+RES -> V1)", fontsize=9)
ax.legend(fontsize=7); ax.set_ylim(0, 0.9)
fig.tight_layout()
fig.savefig("paper/figures/benchmarks.pdf")

# Fig 2: Xrcc4 risk profile
r = json.load(open("results/riskflag_xrcc4.json"))
fig, ax = plt.subplots(figsize=(7, 3))
cuts = [s["cut"] for s in r["top3"]]
# recompute full profile for the figure
import sys; sys.path.insert(0, ".")
from Bio import SeqIO
from tools_riskflag import flag_cutsite
rec = next(SeqIO.parse("data/Xrcc4_mouse.fa", "fasta"))
ctx = str(rec.seq).upper()
xs, ys, spans = [], [], []
for cut in range(250, len(ctx) - 250, 25):
    rep = flag_cutsite(ctx, cut)
    xs.append(cut); ys.append(rep.score); spans.append(rep.max_deletion_span)
ax.plot(xs, ys, lw=1.5, color="#aa3333", label="risk score")
ax.axhline(0.5, ls="--", lw=0.8, color="k"); ax.text(xs[0], 0.51, "HIGH threshold", fontsize=7)
ax.axhline(0.25, ls=":", lw=0.8, color="k"); ax.text(xs[0], 0.26, "MODERATE threshold", fontsize=7)
ax.set_xlabel("cut position on NM_028012.4 (mouse Xrcc4 mRNA)")
ax.set_ylabel("large-deletion risk score")
ax.set_title("Tool 8: risk profile along the Kosicki-2018 Xrcc4 locus", fontsize=9)
fig.tight_layout()
fig.savefig("paper/figures/xrcc4_risk.pdf")
print("figures written")
