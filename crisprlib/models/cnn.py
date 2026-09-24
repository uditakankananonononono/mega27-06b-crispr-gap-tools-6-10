"""CNN architectures for guide and guide:target-pair scoring."""
from __future__ import annotations

import torch
import torch.nn as nn


class GuideCNN(nn.Module):
    """1D CNN over one-hot encoded guide sequence.

    Multi-width kernels capture position-specific motifs (seed vs PAM-distal).
    Input: (B, 4, L). Output: (B,) regression or logit score.
    """

    def __init__(self, seq_len: int = 23, in_ch: int = 4,
                 kernel_widths: tuple[int, ...] = (3, 5, 7), filters: int = 40,
                 dropout: float = 0.2, binary: bool = False):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, filters, k, padding=k // 2),
                nn.ReLU(),
                nn.BatchNorm1d(filters),
            ) for k in kernel_widths
        ])
        flat = filters * len(kernel_widths) * seq_len
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flat, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.binary = binary

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [c(x) for c in self.convs]
        h = torch.cat(feats, dim=1)
        out = self.head(h).squeeze(-1)
        return torch.sigmoid(out) if self.binary else out


class PairCNN(nn.Module):
    """CNN over an aligned guide:target pair (one-hot guide + one-hot target +
    mismatch channel) for off-target classification."""

    def __init__(self, seq_len: int = 23, kernel_widths: tuple[int, ...] = (3, 5),
                 filters: int = 32, dropout: float = 0.3):
        super().__init__()
        in_ch = 9  # 4 guide + 4 target + 1 mismatch
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_ch, filters, k, padding=k // 2),
                          nn.ReLU(), nn.BatchNorm1d(filters))
            for k in kernel_widths
        ])
        flat = filters * len(kernel_widths) * seq_len
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(flat, 96), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(96, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.cat([c(x) for c in self.convs], dim=1)
        return torch.sigmoid(self.head(h).squeeze(-1))
