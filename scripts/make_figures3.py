"""Figure: clean Azimuth-protocol ladder vs same-split Rule Set 2 replication."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("results/beat_azimuth_v1.json") as f:
    v1 = json.load(f)
with open("results/beat_azimuth_v3.json") as f:
    v3 = json.load(f)
with open("results/beat_azimuth_v5.json") as f:
    v5 = json.load(f)

genes = sorted(v1["cnn"]["per_gene"].keys())
cnn = [v1["cnn"]["per_gene"][g] for g in genes]
ours3 = [v3["per_gene"][g] for g in genes]
leader = [v5["leader_rule_set_2_replication"]["per_gene"][g] for g in genes]
chal5 = [v5["challenger_ours"]["per_gene"][g] for g in genes]

x = np.arange(len(genes)); w = 0.2
fig, ax = plt.subplots(figsize=(9.5, 4))
ax.bar(x - 1.5*w, cnn, w, label="GuideCNN (0.286)", color="#aaaaaa")
ax.bar(x - 0.5*w, ours3, w, label="Our best GBT v3 (0.480)", color="#c44e52")
ax.bar(x + 0.5*w, chal5, w, label="Deep-stack tuned v5 (0.370)", color="#8172b3")
ax.bar(x + 1.5*w, leader, w, label="Rule Set 2 replication (0.569)", color="#55a868")
ax.axhline(0.569, color="#55a868", ls="--", lw=1.0, alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels(genes, fontsize=8)
ax.set_ylabel("Per-gene Spearman")
ax.set_title("Clean protocol (train RES human, test V1 mouse): same-split Rule Set 2 replication remains unbeaten")
ax.legend(loc="upper right", fontsize=7.5)
fig.tight_layout()
fig.savefig("paper/figures/azimuth_break.pdf")
print("figure written")
