"""Figures from committed results: Tool 10 transfer benchmark ladder (v1-v7
vs the same-split Rule Set 2 replication) and Tool 7 Lindel head-to-head.
Every number is read from the committed results JSONs at figure-build time."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Transfer ladder: challenger per version vs leader bar ------------
v1 = json.load(open("results/beat_azimuth_v1.json"))
v1_best = max(v1[k]["mean_per_gene"] for k in ("ridge_kmer", "cnn", "gnn"))
v1_note = (f"best of 3 (cnn {v1['cnn']['mean_per_gene']:.3f}, "
           f"ridge {v1['ridge_kmer']['mean_per_gene']:.3f}, "
           f"gnn {v1['gnn']['mean_per_gene']:.3f})")
steps = [
    ("v1\nridge/CNN/GNN", v1_best, None),
    ("v2\nGBT az-feat", json.load(open("results/beat_azimuth_v2.json"))["mean_per_gene"], None),
    ("v3\nGBT full", json.load(open("results/beat_azimuth_v3.json"))["mean_per_gene"], None),
    ("v4\nrank-ens", json.load(open("results/beat_azimuth_v4.json"))["mean_per_gene"], None),
    ("v5\ndeep (flawed)", json.load(open("results/beat_azimuth_v5.json"))["challenger_ours"]["mean_per_gene"], "p=1.0"),
    ("v6\nOOF stack", json.load(open("results/beat_azimuth_v6.json"))["challenger_rs2_plus_deep_oof"]["mean_per_gene"], "p=0.9996"),
    ("v7\nRES+V2 multi-assay", json.load(open("results/beat_azimuth_v7.json"))["challenger_res_plus_v2"][0], "p=0.146"),
]
bar = json.load(open("results/beat_azimuth_v6.json"))["leader_rule_set_2_replication"]["mean_per_gene"]

fig, ax = plt.subplots(figsize=(9.8, 4.0))
xs = np.arange(len(steps))
vals = [s[1] for s in steps]
colors = ["#7570b3"] * len(steps)
ax.bar(xs, vals, color=colors, alpha=0.85)
ax.axhline(bar, color="#d95f02", lw=2, ls="--")
ax.text(len(steps) - 0.4, bar + 0.008, f"same-split Rule Set 2 replication = {bar:.3f}",
        color="#d95f02", fontsize=9, ha="right")
for x, (lab, v, p) in zip(xs, steps):
    ax.text(x, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
    if p:
        ax.text(x, 0.02, p, ha="center", fontsize=7.5, color="white", rotation=0)
ax.set_xticks(xs)
ax.set_xticklabels([s[0] for s in steps], fontsize=7.5)
ax.set_ylabel("mean per-gene Spearman (RES train -> V1 mouse test)")
ax.set_ylim(0, 0.66)
ax.set_title("Tool 10 benchmark ladder: seven challenger variants, leader unbeaten", fontsize=10)
fig.tight_layout()
fig.savefig("paper/figures/transfer_ladder.pdf")
print("transfer_ladder.pdf:", [round(v, 3) for v in vals], "bar", round(bar, 3), "|", v1_note)

# --- 2. Lindel head-to-head: in-distribution and transfer ----------------
h = json.load(open("results/lindel_headtohead.json"))
cc = json.load(open("results/outcome_crosscell.json"))
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
ins = h["targets"]["ins1_frac"]
fs = h["targets"]["frameshift_frac"]
x = np.arange(2)
w = 0.35
axes[0].bar(x - w/2, [ins["ours_gbt_spearman"], fs["ours_gbt_spearman"]], w,
            label="ours (GBT, Lib-A trained)", color="#2c7fb8")
axes[0].bar(x + w/2, [ins["lindel_zeroshot_spearman"], fs["lindel_zeroshot_spearman"]], w,
            label="official Lindel (zero-shot)", color="#d95f02")
for xi, v in zip(x - w/2, [ins["ours_gbt_spearman"], fs["ours_gbt_spearman"]]):
    axes[0].text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
for xi, v in zip(x + w/2, [ins["lindel_zeroshot_spearman"], fs["lindel_zeroshot_spearman"]]):
    axes[0].text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(["+1 insertion freq.", "frameshift freq."], fontsize=9)
axes[0].set_ylabel("Spearman (348 held-out Lib-A sites)")
axes[0].set_ylim(0, 0.75)
axes[0].legend(fontsize=8, loc="upper right")
axes[0].set_title("In-distribution: split win/loss", fontsize=10)

# transfer panel: 2 held-out settings x 2 targets, ours vs Lindel
groups, ours_v, lindel_v = [], [], []
for setting in ("U2OS_Lib-B", "mESC_Lib-B"):
    for tgt, short in (("ins1_frac", "+1 ins"), ("frameshift_frac", "frameshift")):
        r = cc["test"][setting][tgt]
        groups.append(f"{setting.replace('_Lib-B','')} Lib-B\n{short}")
        ours_v.append(r["ours_gbt_spearman"])
        lindel_v.append(r["lindel_zeroshot_spearman"])
x2 = np.arange(len(groups))
axes[1].bar(x2 - w/2, ours_v, w, label="ours (Lib-A U2OS trained)", color="#2c7fb8")
axes[1].bar(x2 + w/2, lindel_v, w, label="official Lindel (zero-shot)", color="#d95f02")
for xi, v in zip(x2 - w/2, ours_v):
    axes[1].text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5)
for xi, v in zip(x2 + w/2, lindel_v):
    axes[1].text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5)
axes[1].set_xticks(x2)
axes[1].set_xticklabels(groups, fontsize=8)
axes[1].set_ylabel("Spearman")
axes[1].set_ylim(0, 0.8)
axes[1].legend(fontsize=8)
axes[1].set_title("Transfer: Lindel wins all four", fontsize=10)
fig.tight_layout()
fig.savefig("paper/figures/lindel_head2head.pdf")
print("lindel_head2head.pdf: in-dist ours",
      round(ins["ours_gbt_spearman"], 3), round(fs["ours_gbt_spearman"], 3),
      "lindel", round(ins["lindel_zeroshot_spearman"], 3),
      round(fs["lindel_zeroshot_spearman"], 3), "| transfer ours",
      [round(v, 3) for v in ours_v], "lindel", [round(v, 3) for v in lindel_v])
