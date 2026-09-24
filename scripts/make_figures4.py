"""Figure: real-guide duplex risk audit."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results/realguide_risk.json") as f:
    r = json.load(f)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
labels = ["Random spacers\n(7,140 pairs)", "Cross-gene published\npairs (30,000)",
          "Same-gene published\npairs (30,000)"]
vals = [r["random_spacer_baseline"], r["cross_gene_flag_rate"], r["same_gene_flag_rate"]]
bars = ax.bar(labels, vals, color=["#aaaaaa", "#4c72b0", "#c44e52"])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.004, f"{v*100:.1f}%", ha="center")
ax.set_ylabel("Fraction duplex-flagged (MFE <= -12 kcal/mol)")
ax.set_title("Same-gene duplex enrichment in published guide sets (4,692 real guides)")
ax.set_ylim(0, 0.27)
fig.tight_layout()
fig.savefig("paper/figures/realguide_risk.pdf")
print("ok")
