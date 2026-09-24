"""Figure: benchmark break on the Azimuth cross-dataset protocol."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("results/beat_azimuth_v1.json") as f:
    res = json.load(f)

genes = sorted(res["cnn"]["per_gene"].keys())
cnn = [res["cnn"]["per_gene"][g] for g in genes]
ridge = [res["ridge_kmer"]["per_gene"][g] for g in genes]
gnn = [res["gnn"]["per_gene"][g] for g in genes]

x = np.arange(len(genes)); w = 0.27
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x - w, ridge, w, label="Ridge (3-mer)", color="#aaaaaa")
ax.bar(x, gnn, w, label="SequenceGCN", color="#8172b3")
ax.bar(x + w, cnn, w, label="GuideCNN (ours)", color="#c44e52")
ax.axhline(0.52, color="black", ls="--", lw=1.2,
           label="Published Rule Set 2 on FC (0.52)")
ax.set_xticks(x); ax.set_xticklabels(genes)
ax.set_ylabel("Per-gene Spearman")
ax.set_title("Cross-dataset protocol (train RES, test V1/FC): GuideCNN vs published Rule Set 2")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("paper/figures/azimuth_break.pdf")
print("figure written")
