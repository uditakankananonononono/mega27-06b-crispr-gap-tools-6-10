"""Doench 2016 human (FC+RES) and mouse (V1) guide-activity loaders."""
from __future__ import annotations

import pandas as pd


def load_human(path: str = "data/FC_plus_RES_withPredictions.csv") -> pd.DataFrame:
    """FC+RES: 5,310 human guides; target = score_drug_gene_rank (0..1)."""
    df = pd.read_csv(path, index_col=0)
    df = df.rename(columns={"30mer": "context", "score_drug_gene_rank": "activity"})
    df = df.dropna(subset=["context", "activity"])
    df["context"] = df["context"].str.upper()
    df = df[df["context"].str.len() == 30]
    return df[["context", "activity", "Target gene", "drug"]]


def load_mouse(path: str = "data/V1_suppl_data.txt") -> pd.DataFrame:
    """V1: 2,144 mouse guides; 34-nt extended context trimmed to azimuth 30-mer;
    target = Percent Rank (0..1)."""
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={"Extended Spacer(NNNN[20nt]NGGNNNNNNN)": "ext"})
    df["context"] = df["ext"].str.upper().str[:30]
    df["activity"] = pd.to_numeric(df["Percent Rank"], errors="coerce")
    df = df.dropna(subset=["context", "activity"])
    df = df[df["context"].str.len() == 30]
    return df[["context", "activity", "Gene Symbol"]]
