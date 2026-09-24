"""Figures from committed results: Tool 8 panel risk distribution,
off-target per-study replication, classic scorers on the RES->V1 split."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Tool 8 panel: locus risk distribution + top loci -----------------
p = json.load(open("results/riskflag_panel.json"))
recs = [r for r in p["records"] if r.get("risk_mean") is not None]
means = np.array([r["risk_mean"] for r in recs])
maxs = np.array([r["risk_max"] for r in recs])
top = sorted(recs, key=lambda r: -r["risk_mean"])[:15]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
axes[0].hist(means, bins=24, color="#2c7fb8", alpha=0.85, label="locus mean risk")
axes[0].hist(maxs, bins=24, color="#d95f02", alpha=0.6, label="locus max site risk")
axes[0].set_xlabel("large-deletion risk score")
axes[0].set_ylabel("loci")
axes[0].axvline(np.mean(means), color="#2c7fb8", ls="--", lw=1)
axes[0].text(np.mean(means) + 0.005, axes[0].get_ylim()[1] * 0.9,
             f"mean {np.mean(means):.3f}", fontsize=8, color="#2c7fb8")
axes[0].legend(fontsize=8)
axes[0].set_title(f"Tool 8 panel: {len(recs)} loci, "
                  f"{sum(r['n_cutsites_scored'] for r in recs)} cut sites", fontsize=10)

labels = [f"{r['gene']} ({r['organism'][:3]})" for r in top][::-1]
vals = [r["risk_mean"] for r in top][::-1]
axes[1].barh(np.arange(len(top)), vals, color="#2c7fb8")
axes[1].set_yticks(np.arange(len(top)))
axes[1].set_yticklabels(labels, fontsize=7)
for i, v in enumerate(vals):
    axes[1].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7)
axes[1].set_xlabel("mean locus risk")
axes[1].set_title("Top 15 loci by mean risk", fontsize=10)
axes[1].set_xlim(0, max(vals) * 1.25)
fig.tight_layout()
fig.savefig("paper/figures/panel_risk.pdf")
plt.close(fig)

# --- 2. Off-target per-study replication ---------------------------------
o = json.load(open("results/offtarget_scorers.json"))
studies = list(o["studies"].keys())
hsu = [o["studies"][s]["auc_hsu_style"] for s in studies]
cfd = [o["studies"][s]["auc_cfd_style"] for s in studies]
x = np.arange(len(studies)); w = 0.36
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(x - w/2, hsu, w, label="Hsu-style (position-mismatch logistic)", color="#7570b3")
ax.bar(x + w/2, cfd, w, label="CFD-style (position x base table)", color="#1b9e77")
ax.axhline(0.5, color="k", lw=0.8, ls=":")
ax.axhline(o["mean_auc"]["hsu_style"], color="#7570b3", ls="--", lw=1)
ax.axhline(o["mean_auc"]["cfd_style"], color="#1b9e77", ls="--", lw=1)
ax.text(-0.45, o["mean_auc"]["hsu_style"] - 0.022,
        f"mean {o['mean_auc']['hsu_style']:.3f}", fontsize=7, color="#7570b3", ha="left")
ax.text(-0.45, o["mean_auc"]["cfd_style"] + 0.008,
        f"mean {o['mean_auc']['cfd_style']:.3f}", fontsize=7, color="#1b9e77", ha="left")
ax.set_xticks(x)
ax.set_xticklabels([f"{s}\n(n={o['studies'][s]['n']:,})" for s in studies], fontsize=8)
ax.set_ylabel("LOSO AUC (logistic replication)")
ax.set_title("Off-target scorer replications on crisprSQL (16,472 pairs, 7 studies)",
             fontsize=10)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig("paper/figures/offtarget_perstudy.pdf")
plt.close(fig)

# --- 3. Classic scorers on the clean RES->V1 split ------------------------
c = json.load(open("results/classic_scorers.json"))
c2 = json.load(open("results/classic_scorers2.json"))
v7 = json.load(open("results/beat_azimuth_v7.json"))
v6 = json.load(open("results/beat_azimuth_v6.json"))
names = ["CRISPRscan-style\n(6-mer, 2013)", "WU-CRISPR-style\n(struct., 2015)",
         "Chari-style\n(motif, 2015)", "Rule Set 1-style\n(logistic, 2014)",
         "SSC-style\n(linear, 2014)", "OOF deep stack\n(ours, v6)",
         "multi-assay\nRES+V2 (ours, v7)", "Rule Set 2\nreplication (bar)"]
vals = [c["crisprscan_style_6mer"][0], c2["wu_crispr_style"][0],
        c2["chari_style_logistic"][0], c["rs1_style_logistic"][0],
        c["ssc_style_linear"][0], v6["challenger_rs2_plus_deep_oof"]["mean_per_gene"],
        v7["challenger_res_plus_v2"][0], v7["leader_res_only"][0]]
colors = ["#999999"] * 5 + ["#2c7fb8", "#2c7fb8", "#d95f02"]
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.bar(np.arange(len(vals)), vals, color=colors)
for i, v in enumerate(vals):
    ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(np.arange(len(vals)))
ax.set_xticklabels(names, fontsize=7.5)
ax.set_ylabel("mean per-gene Spearman (V1 mouse test)")
ax.set_title("Cross-species transfer: classic scorers, our challengers, and the bar",
             fontsize=10)
ax.set_ylim(0, max(vals) * 1.2)
fig.tight_layout()
fig.savefig("paper/figures/classic_scorers.pdf")
plt.close(fig)
print("wrote panel_risk.pdf offtarget_perstudy.pdf classic_scorers.pdf")
