"""Figure: Tool 7 transfer matrix (ours vs official Lindel) from committed
results/lindel_headtohead.json + results/outcome_crosscell.json."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

h2h = json.load(open("results/lindel_headtohead.json"))
cc = json.load(open("results/outcome_crosscell.json"))

rows = [
    ("Lib-A held-out\n(348)", h2h["targets"]["ins1_frac"]["ours_gbt_spearman"],
     h2h["targets"]["ins1_frac"]["lindel_zeroshot_spearman"],
     h2h["targets"]["frameshift_frac"]["ours_gbt_spearman"],
     h2h["targets"]["frameshift_frac"]["lindel_zeroshot_spearman"]),
    ("U2OS Lib-B\n(1,510)", cc["test"]["U2OS_Lib-B"]["ins1_frac"]["ours_gbt_spearman"],
     cc["test"]["U2OS_Lib-B"]["ins1_frac"]["lindel_zeroshot_spearman"],
     cc["test"]["U2OS_Lib-B"]["frameshift_frac"]["ours_gbt_spearman"],
     cc["test"]["U2OS_Lib-B"]["frameshift_frac"]["lindel_zeroshot_spearman"]),
    ("mESC Lib-B\n(1,953)", cc["test"]["mESC_Lib-B"]["ins1_frac"]["ours_gbt_spearman"],
     cc["test"]["mESC_Lib-B"]["ins1_frac"]["lindel_zeroshot_spearman"],
     cc["test"]["mESC_Lib-B"]["frameshift_frac"]["ours_gbt_spearman"],
     cc["test"]["mESC_Lib-B"]["frameshift_frac"]["lindel_zeroshot_spearman"]),
]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
x = np.arange(3); w = 0.36
for ax, (oi, li, title) in zip(axes, [(1, 2, "+1 insertion rate"),
                                      (3, 4, "frameshift frequency")]):
    ours = [r[oi] for r in rows]; lin = [r[li] for r in rows]
    ax.bar(x - w/2, ours, w, label="ours (GBT positional)", color="#2c7fb8")
    ax.bar(x + w/2, lin, w, label="Lindel official (zero-shot)", color="#d95f02")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_title(title, fontsize=10)
    for xi, v in zip(x - w/2, ours):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    for xi, v in zip(x + w/2, lin):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
axes[0].set_ylabel("Spearman vs observed")
axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Repair-outcome prediction: in-distribution win, transfer losses (real data)",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("paper/figures/outcome_transfer_matrix.pdf")
print("wrote paper/figures/outcome_transfer_matrix.pdf")
