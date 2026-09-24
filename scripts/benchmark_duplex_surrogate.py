"""Benchmark: CNN and GNN surrogates for ViennaRNA self-fold MFE prediction.

Teacher/ground truth: ViennaRNA RNA.fold MFE on 2,000 random 20-nt spacers
(realistic GC 30-70%). Held-out 20%. Baselines: ridge regression on 3-mer
counts. Run live (not part of CI); writes results/duplex_surrogate.json.
"""
import json
import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from crisprlib.featurize import kmer_counts, one_hot
from crisprlib.models import GuideCNN, SequenceGCN, kmer_graph
from crisprlib.train import predict, train_regressor
from crisprlib.benchmark import regression_metrics
from tools_multiplex.duplex import self_fold_mfe

rng = np.random.default_rng(7)
N = 2000
seqs = []
while len(seqs) < N:
    gc_target = rng.uniform(0.3, 0.7)
    p = np.array([(1 - gc_target) / 2, gc_target / 2, gc_target / 2, (1 - gc_target) / 2])
    seqs.append("".join(rng.choice(list("ACGT"), size=20, p=p)))

t0 = time.time()
y = np.array([self_fold_mfe(s) for s in seqs], dtype=np.float32)
print(f"ViennaRNA fold on {N} spacers: {time.time() - t0:.1f}s; "
      f"MFE range {y.min():.2f}..{y.max():.2f}")

X_oh = np.stack([one_hot(s) for s in seqs]).transpose(0, 2, 1)  # (N,4,20)
X_km = np.stack([kmer_counts(s, 3) for s in seqs])
X_graph = [kmer_graph(s, k=3) for s in seqs]

idx = np.arange(N)
tr, te = train_test_split(idx, test_size=0.2, random_state=0)

# ridge baseline on kmers
ridge = Ridge(alpha=1.0).fit(X_km[tr], y[tr])
p_ridge = ridge.predict(X_km[te])

# CNN surrogate
cnn = GuideCNN(seq_len=20, filters=32, kernel_widths=(3, 5, 7), dropout=0.2)
cnn, _ = train_regressor(cnn, X_oh[tr], y[tr], X_oh[te], y[te], epochs=50, lr=2e-3, seed=0)
p_cnn = predict(cnn, X_oh[te])

# GNN surrogate
gnn = SequenceGCN(k=3, hidden=48, layers=2, dropout=0.2)
gnn, _ = train_regressor(gnn, [X_graph[i] for i in tr], y[tr],
                         [X_graph[i] for i in te], y[te],
                         epochs=50, lr=2e-3, seed=0, gnn=True)
p_gnn = predict(gnn, [X_graph[i] for i in te], gnn=True)

res = {
    "n_train": int(len(tr)), "n_test": int(len(te)),
    "teacher": "ViennaRNA 2.7.2 RNA.fold MFE (kcal/mol)",
    "ridge_kmer": regression_metrics(y[te], p_ridge),
    "cnn": regression_metrics(y[te], p_cnn),
    "gnn": regression_metrics(y[te], p_gnn),
}
print(json.dumps(res, indent=2))
json.dump(res, open("results/duplex_surrogate.json", "w"), indent=2)
