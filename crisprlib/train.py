"""Deterministic training loops for CNN/GNN scorers (CPU-sized)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def train_regressor(model, x_tr, y_tr, x_va=None, y_va=None, epochs=40, lr=1e-3,
                    batch=64, seed=0, weight_decay=1e-4, verbose=False,
                    gnn: bool = False):
    """MSE training with early stopping on validation loss (or train loss if no val).

    gnn=True expects x as list of (node_feat, adj) tuples; otherwise x is a
    (N, C, L) float tensor. Returns (model, history dict).
    """
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    hist = {"train": [], "val": []}
    best_val, best_state, patience, wait = float("inf"), None, 8, 0
    n = len(y_tr)
    idx_all = np.arange(n)
    for ep in range(epochs):
        model.train()
        rng = np.random.default_rng(seed + ep)
        rng.shuffle(idx_all)
        tot, cnt = 0.0, 0
        for s in range(0, n, batch):
            bidx = idx_all[s:s + batch]
            opt.zero_grad()
            if gnn:
                loss = _gnn_batch_loss(model, loss_fn, x_tr, y_tr, bidx)
            else:
                xb = torch.as_tensor(x_tr[bidx])
                yb = torch.as_tensor(np.asarray(y_tr)[bidx], dtype=torch.float32)
                loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(bidx)
            cnt += len(bidx)
        tr = tot / max(cnt, 1)
        hist["train"].append(tr)
        if x_va is not None:
            vl = evaluate_mse(model, x_va, y_va, gnn=gnn)
            hist["val"].append(vl)
            if vl < best_val - 1e-5:
                best_val, wait = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
        if verbose:
            print(f"epoch {ep}: train {tr:.4f}" + (f" val {hist['val'][-1]:.4f}" if x_va is not None else ""))
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


def _gnn_batch_loss(model, loss_fn, xs, ys, bidx):
    xs_b = [xs[i] for i in bidx]
    nmax = max(x[0].shape[0] for x in xs_b)
    fdim = xs_b[0][0].shape[1]
    xb = torch.zeros(len(xs_b), nmax, fdim)
    ab = torch.zeros(len(xs_b), nmax, nmax)
    for i, (xf, af) in enumerate(xs_b):
        ni = xf.shape[0]
        xb[i, :ni] = torch.as_tensor(xf)
        ab[i, :ni, :ni] = torch.as_tensor(af)
    yb = torch.as_tensor(np.asarray(ys)[bidx], dtype=torch.float32)
    return loss_fn(model(xb, ab), yb)


@torch.no_grad()
def predict(model, x, gnn: bool = False, batch: int = 128):
    model.eval()
    if gnn:
        outs = []
        for s in range(0, len(x), batch):
            xs_b = x[s:s + batch]
            nmax = max(t[0].shape[0] for t in xs_b)
            fdim = xs_b[0][0].shape[1]
            xb = torch.zeros(len(xs_b), nmax, fdim)
            ab = torch.zeros(len(xs_b), nmax, nmax)
            for i, (xf, af) in enumerate(xs_b):
                ni = xf.shape[0]
                xb[i, :ni] = torch.as_tensor(xf)
                ab[i, :ni, :ni] = torch.as_tensor(af)
            outs.append(model(xb, ab).numpy())
        return np.concatenate(outs)
    x = torch.as_tensor(np.asarray(x))
    return torch.cat([model(x[s:s + batch]) for s in range(0, len(x), batch)]).numpy()


@torch.no_grad()
def evaluate_mse(model, x, y, gnn: bool = False):
    p = predict(model, x, gnn=gnn)
    y = np.asarray(y, dtype=np.float32)
    return float(np.mean((p - y) ** 2))
