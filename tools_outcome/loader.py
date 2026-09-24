"""inDelphi U2OS LibA loader: event-level figshare CSV joined to target contexts.

Data provenance:
- Event counts: figshare 6837956, file U2OS_+_LibA_postCas9_rep1.csv
  (Shen et al. 2018, Nature 563:646, doi:10.1038/s41586-018-0686-x).
- Target contexts (57-nt, cut between positions 28/29 0-based):
  maxwshen/inDelphi-dataprocessinganalysis, data-libprocessing/targets-libA.txt.
The event CSV's `_Experiment` column is the 0-based index into targets-libA.txt.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

CUT_POS = 27  # empirical: Lib-A/B guides align at ctx[10:30] (grna-lib*.txt
# exact match in 1996/2000 Lib-A targets), PAM ctx[30:33], so the
# canonical Cas9 cut is between indices 26/27


@dataclass
class SiteOutcomes:
    name: str
    context: str               # 57-nt target context
    total_reads: float
    frameshift_frac: float
    ins1_frac: float           # +1 insertion fraction of all reads
    mh_del_frac: float         # microhomology deletion fraction of all deletions
    top_outcome_frac: float    # precision: most frequent single outcome
    mean_del_len: float


def load_targets(path: str) -> list[str]:
    seqs = []
    with open(path) as f:
        for line in f:
            s = line.strip().upper()
            if s and set(s) <= set("ACGT"):
                seqs.append(s)
    return seqs


def aggregate_events(events_csv: str, targets: list[str]) -> list[SiteOutcomes]:
    df = pd.read_csv(events_csv, index_col=0)
    df = df.dropna(subset=["_Experiment"])
    df["_Experiment"] = df["_Experiment"].astype(int)
    out: list[SiteOutcomes] = []
    for exp, grp in df.groupby("_Experiment"):
        if exp >= len(targets):
            continue
        ctx = targets[exp]
        total = float(grp["Count"].sum())
        # CRISPR-induced edited reads only (exclude notcrispr / notatcut /
        # pcr_recombination / other / wildtype) - the outcome distribution is
        # over Cas9-induced indels.
        indel_mask = grp["Category"].isin(["del", "ins", "combination_indel"])
        edited = float(grp.loc[indel_mask, "Count"].sum())
        if total < 50 or edited < 100:
            continue
        dels = grp[grp["Category"] == "del"]
        inss = grp[grp["Category"] == "ins"]
        n_del = float(dels["Count"].sum())
        n_mh = float(dels.loc[dels["Microhomology-Based"] == "yes", "Count"].sum())
        ins1 = float(inss.loc[inss["Length"] == 1, "Count"].sum())
        fs = 0.0
        if edited > 0:
            dl = dels.dropna(subset=["Length"])
            fs_reads = float(dl.loc[(dl["Length"] % 3) != 0, "Count"].sum()) + \
                       float(inss.loc[(inss["Length"] % 3) != 0, "Count"].sum())
            fs = fs_reads / edited
        nonwt = grp.loc[~grp["Category"].isin(["wildtype", "pcr_recombination"])]
        top = float(nonwt["Count"].max()) if len(nonwt) else 0.0
        out.append(SiteOutcomes(
            name=f"LibA-{exp}", context=ctx, total_reads=total,
            frameshift_frac=fs,
            ins1_frac=ins1 / total,
            mh_del_frac=(n_mh / n_del) if n_del > 0 else 0.0,
            top_outcome_frac=(top / edited) if edited > 0 else 0.0,
            mean_del_len=float((dels["Length"] * dels["Count"]).sum() / n_del) if n_del > 0 else 0.0,
        ))
    return out
