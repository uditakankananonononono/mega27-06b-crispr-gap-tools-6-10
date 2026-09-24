"""Gap 10 benchmark: cross-species transfer of guide-efficacy models.

Train on human (Doench FC+RES), evaluate:
  (a) held-out human, (b) full mouse (V1) transfer.
Models: ridge (kmer), CNN (one-hot 30-mer), GNN (kmer graph).
Plus species-shift diagnostics (GC delta, kmer JS divergence).
Live run (not CI); writes results/transfer.json.
"""
import json

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from crisprlib.benchmark import regression_metrics
from crisprlib.featurize import kmer_counts, one_hot
from crisprlib.models import GuideCNN, SequenceGCN, kmer_graph
from crisprlib.train import predict, train_regressor
from tools_transfer import load_human, load_mouse
from tools_transfer.divergence import species_shift

hu = load_human()
mo = load_mouse()
print(f"human {len(hu)} guides, mouse {len(mo)} guides")

X_hu = np.stack([one_hot(s, 30) for s in hu["context"]]).transpose(0, 2, 1)
y_hu = hu["activity"].to_numpy(np.float32)
X_mo = np.stack([one_hot(s, 30) for s in mo["context"]]).transpose(0, 2, 1)
y_mo = mo["activity"].to_numpy(np.float32)
K_hu = np.stack([kmer_counts(s, 3) for s in hu["context"]])
K_mo = np.stack([kmer_counts(s, 3) for s in mo["context"]])
G_hu = [kmer_graph(s, k=3) for s in hu["context"]]
G_mo = [kmer_graph(s, k=3) for s in mo["context"]]

tr, te = train_test_split(np.arange(len(hu)), test_size=0.2, random_state=0)
res = {"n_human_train": int(len(tr)), "n_human_test": int(len(te)), "n_mouse": len(mo)}

ridge = Ridge(alpha=1.0).fit(K_hu[tr], y_hu[tr])
res["ridge_kmer"] = {"human_heldout": regression_metrics(y_hu[te], ridge.predict(K_hu[te])),
                     "mouse_transfer": regression_metrics(y_mo, ridge.predict(K_mo))}
print("ridge", json.dumps(res["ridge_kmer"]))

cnn = GuideCNN(seq_len=30, filters=32, kernel_widths=(3, 5, 7))
cnn, _ = train_regressor(cnn, X_hu[tr], y_hu[tr], X_hu[te], y_hu[te], epochs=60, lr=2e-3, seed=0)
res["cnn"] = {"human_heldout": regression_metrics(y_hu[te], predict(cnn, X_hu[te])),
              "mouse_transfer": regression_metrics(y_mo, predict(cnn, X_mo))}
print("cnn", json.dumps(res["cnn"]))

gnn = SequenceGCN(k=3, hidden=48, layers=2)
gnn, _ = train_regressor(gnn, [G_hu[i] for i in tr], y_hu[tr],
                         [G_hu[i] for i in te], y_hu[te], epochs=60, lr=2e-3, seed=0, gnn=True)
res["gnn"] = {"human_heldout": regression_metrics(y_hu[te], predict(gnn, [G_hu[i] for i in te], gnn=True)),
              "mouse_transfer": regression_metrics(y_mo, predict(gnn, G_mo, gnn=True))}
print("gnn", json.dumps(res["gnn"]))

res["species_shift"] = species_shift(list(hu["context"]), list(mo["context"]))
print("shift", json.dumps(res["species_shift"]))
for m in ("ridge_kmer", "cnn", "gnn"):
    h = res[m]["human_heldout"]["spearman"]
    mm = res[m]["mouse_transfer"]["spearman"]
    res[m]["transfer_drop"] = round(h - mm, 4)
json.dump(res, open("results/transfer.json", "w"), indent=2)
