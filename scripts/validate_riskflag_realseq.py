"""Validate the risk flagger on the three Kosicki-2018 loci (real sequences).

Mouse Xrcc4 (NM_028012.4), Piga (NM_011081.4), Cd9 (NM_007657.4) mRNAs fetched
live from NCBI - the three loci where Kosicki et al. 2018 measured kilobase-
scale deletions in mESCs. Every 20-nt window is flagged as a hypothetical cut
site. Live (not CI); writes results/riskflag_loci.json.
"""
import json

from Bio import SeqIO
from tools_riskflag import flag_cutsite

out = {}
for locus, path in [("Xrcc4", "data/Xrcc4_mouse.fa"),
                    ("Piga", "data/Piga_mouse.fa"),
                    ("Cd9", "data/Cd9_mouse.fa")]:
    rec = next(SeqIO.parse(path, "fasta"))
    ctx = str(rec.seq).upper()
    scores = []
    for cut in range(250, len(ctx) - 250, 25):
        r = flag_cutsite(ctx, cut)
        scores.append({"cut": cut, "score": r.score, "flag": r.flag,
                       "max_span": r.max_deletion_span})
    out[locus] = {"accession": rec.id, "len": len(ctx),
                  "n_scanned": len(scores),
                  "n_high": sum(1 for s in scores if s["flag"] == "HIGH"),
                  "n_moderate": sum(1 for s in scores if s["flag"] == "MODERATE"),
                  "max_span_overall": max((s["max_span"] for s in scores), default=0),
                  "mean_score": round(sum(s["score"] for s in scores) / max(len(scores), 1), 4)}
    print(locus, json.dumps(out[locus]))
json.dump(out, open("results/riskflag_loci.json", "w"), indent=2)
