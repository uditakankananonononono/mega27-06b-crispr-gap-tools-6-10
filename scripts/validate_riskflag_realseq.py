"""Validate the risk flagger on a real Kosicki-2018 locus sequence.

Mouse Xrcc4 mRNA (RefSeq, accession resolved via NCBI esearch) - one of the three loci where Kosicki et al.
2018 measured kilobase-scale deletions in mESCs. We flag every 20-nt window
as a hypothetical cut site and record the score distribution. Live (not CI).
"""
import json

from Bio import SeqIO
from tools_riskflag import flag_cutsite

rec = next(SeqIO.parse("data/Xrcc4_mouse.fa", "fasta"))
ctx = str(rec.seq).upper()
print(f"{rec.id} len {len(ctx)}")
scores = []
for cut in range(250, len(ctx) - 250, 25):
    r = flag_cutsite(ctx, cut)
    scores.append({"cut": cut, "score": r.score, "flag": r.flag,
                   "max_span": r.max_deletion_span})
hi = [s for s in scores if s["flag"] == "HIGH"]
out = {"locus": rec.id, "n_cuts_scanned": len(scores),
       "n_high": len(hi), "n_moderate": sum(1 for s in scores if s["flag"] == "MODERATE"),
       "top3": sorted(scores, key=lambda s: -s["score"])[:3]}
print(json.dumps(out, indent=2))
json.dump(out, open("results/riskflag_xrcc4.json", "w"), indent=2)
