"""Second batch of paper figures: multiplex design rules and cross-species transfer."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results/multiplex_rules.json") as f:
    rules = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
ax = axes[0]
bins = ["GC<0.45", "0.45-0.60", "GC>0.60"]
rates = [rules["rules"]["low_GC_self_fold_rate"],
         rules["rules"]["mid_GC_self_fold_rate"],
         rules["rules"]["high_GC_self_fold_rate"]]
ax.bar(bins, rates, color=["#4c72b0", "#55a868", "#c44e52"])
ax.set_ylabel("Fraction with strong self-fold\n(MFE < -5 kcal/mol)")
ax.set_xlabel("Guide GC-content bin")
ax.set_title("Self-fold sequestration vs GC content")

ax = axes[1]
labels = ["Random pairs\nduplex-flagged", "Guides with >=4-nt\nhomopolymer self-fold"]
vals = [rules["duplex"]["frac_flagged"],
        rules["rules"]["homopolymer4plus_self_fold_rate"]]
ax.bar(labels, vals, color=["#4c72b0", "#8172b3"])
ax.set_ylabel("Fraction")
ax.set_title("Heterodimer prevalence and homopolymer effect")
for i, v in enumerate(vals):
    ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig("paper/figures/multiplex_rules.pdf")

with open("results/transfer.json") as f:
    tr = json.load(f)
fig, ax = plt.subplots(figsize=(6, 3.6))
models = ["Ridge (k-mer)", "GuideCNN", "SequenceGCN"]
heldout = [tr["ridge_kmer"]["human_heldout"]["spearman"],
           tr["cnn"]["human_heldout"]["spearman"],
           tr["gnn"]["human_heldout"]["spearman"]]
transfer = [tr["ridge_kmer"]["mouse_transfer"]["spearman"],
            tr["cnn"]["mouse_transfer"]["spearman"],
            tr["gnn"]["mouse_transfer"]["spearman"]]
x = range(len(models)); w = 0.36
ax.bar([i - w/2 for i in x], heldout, w, label="Human held-out (FC+RES)", color="#4c72b0")
ax.bar([i + w/2 for i in x], transfer, w, label="Mouse transfer (V1)", color="#c44e52")
ax.set_xticks(list(x)); ax.set_xticklabels(models)
ax.set_ylabel("Spearman correlation")
ax.set_title("Human-to-mouse efficacy transfer by model class")
ax.legend()
fig.tight_layout()
fig.savefig("paper/figures/transfer.pdf")
print("figures written")
