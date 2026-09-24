"""Tool 8 broad validation panel: score candidate cut sites across a panel of
RefSeq accessions (each accession = one dataset in the manifest). Resolves
accessions via NCBI esearch (never guessed), fetches mRNA sequence via
efetch, finds NGG PAM cut sites, and flags each with the mechanistic risk
scorer. Resumable: results append to results/riskflag_panel.json.

Usage: python3 scripts/riskflag_refseq_panel.py --limit 40
"""
import argparse
import json
import time
import urllib.parse
import urllib.request

from crisprlib.featurize import gc_content
from tools_riskflag.flagger import flag_cutsite

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PANEL = [
    # (gene, organism) - editing-relevant panel: crisprSQL frequent targets +
    # common editing loci across human and mouse
    ("HBB", "human"), ("EMX1", "human"), ("VEGFA", "human"),
    ("FANCF", "human"), ("RUNX1", "human"), ("HEK293", None),  # placeholder filtered below
    ("AAVS1", None), ("DNMT1", "human"), ("TET2", "human"),
    ("TP53", "human"), ("BRCA1", "human"), ("BRCA2", "human"),
    ("CFTR", "human"), ("DMD", "human"), ("HTT", "human"),
    ("APOE", "human"), ("PCSK9", "human"), ("CCR5", "human"),
    ("IL2RG", "human"), ("RAG1", "human"), ("ALB", "human"),
    ("G6PD", "human"), ("F8", "human"), ("F9", "human"),
    ("HBG1", "human"), ("HBG2", "human"), ("SERPINA1", "human"),
    ("LDLR", "human"), ("MSTN", "human"), ("TTR", "human"),
    ("SOD1", "human"), ("APP", "human"), ("MAPT", "human"),
    ("SNCA", "human"), ("LRRK2", "human"), ("GBA", "human"),
    ("NF1", "human"), ("RB1", "human"), ("PTEN", "human"),
    ("KRAS", "human"), ("BRAF", "human"), ("EGFR", "human"),
    ("MYC", "human"), ("CDKN2A", "human"), ("VHL", "human"),
    ("Xrcc4", "mouse"), ("Piga", "mouse"), ("Cd9", "mouse"),
    ("Rosa26", "mouse"), ("Sox2", "mouse"), ("Oct4", "mouse"),
    ("Nanog", "mouse"), ("Trp53", "mouse"), ("Pten", "mouse"),
    ("Apc", "mouse"), ("Kras", "mouse"), ("Braf", "mouse"),
    ("Dmd", "mouse"), ("Htt", "mouse"), ("Sod1", "mouse"),
    ("App", "mouse"), ("Mapt", "mouse"), ("Snca", "mouse"),
    ("Ldlr", "mouse"), ("Pcsk9", "mouse"), ("Ttr", "mouse"),
    ("Alb", "mouse"), ("F8", "mouse"), ("Cftr", "mouse"),
    ("Apoe", "mouse"), ("Mstn", "mouse"), ("Myo7a", "mouse"),
    ("Pax6", "mouse"), ("Tyr", "mouse"), ("Kit", "mouse"),
    ("Rag1", "mouse"), ("Il2rg", "mouse"), ("Hbb", "mouse"),
    ("Hba", "mouse"), ("Gata1", "mouse"), ("Runx1", "mouse"),
    ("Dnmt1", "mouse"), ("Tet2", "mouse"), ("Ezh2", "mouse"),
]
PANEL = [(g, o) for g, o in PANEL if o]


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "riskflag-panel/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()


def resolve_refseq(gene, organism):
    org = "Homo sapiens" if organism == "human" else "Mus musculus"
    term = f'{gene}[Gene Name] AND {org}[Organism] AND RefSeq[Filter] AND mRNA[Filter]'
    url = EUTILS + "/esearch.fcgi?db=nuccore&retmode=json&retmax=5&term=" + urllib.parse.quote(term)
    data = json.loads(_get(url))
    ids = data.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return None, None
    time.sleep(0.34)
    summ = json.loads(_get(EUTILS + "/esummary.fcgi?db=nuccore&retmode=json&id=" + ",".join(ids)))
    best = None
    for uid in ids:
        acc = summ["result"][uid].get("caption", "")
        if acc.startswith("NM_"):
            return uid, acc
        if best is None:
            best = (uid, acc)
    return best if best else (None, None)


def fetch_seq(uid):
    txt = _get(EUTILS + f"/efetch.fcgi?db=nuccore&id={uid}&rettype=fasta&retmode=text")
    return "".join(l.strip() for l in txt.splitlines() if not l.startswith(">")).upper()


def score_gene(gene, organism):
    uid, acc = resolve_refseq(gene, organism)
    if not uid:
        return {"gene": gene, "organism": organism, "error": "no_refseq"}
    time.sleep(0.34)
    seq = fetch_seq(uid)
    seq = "".join(c for c in seq if c in "ACGT")
    sites, flags = 0, {"LOW": 0, "MODERATE": 0, "HIGH": 0}
    scores = []
    step = max(1, len(seq) // 60)
    for i in range(0, len(seq) - 23, step):
        if seq[i+1:i+3] == "GG":
            rep = flag_cutsite(seq[max(0, i-260):i+280], min(260, i) - 3)
            flags[rep.flag] += 1
            scores.append(rep.score)
            sites += 1
        if sites >= 40:
            break
    return {"gene": gene, "organism": organism, "accession": acc, "uid": uid,
            "seq_len": len(seq), "gc": gc_content(seq) if seq else None,
            "n_cutsites_scored": sites,
            "flag_counts": flags,
            "risk_mean": float(sum(scores) / len(scores)) if scores else None,
            "risk_max": float(max(scores)) if scores else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", default="results/riskflag_panel.json")
    args = ap.parse_args()
    try:
        done = {r["gene"] + r["organism"] for r in json.load(open(args.out))["records"]}
        records = json.load(open(args.out))["records"]
    except Exception:
        done, records = set(), []
    n_new = 0
    for gene, org in PANEL:
        if gene + org in done or n_new >= args.limit:
            continue
        try:
            rec = score_gene(gene, org)
        except Exception as e:
            rec = {"gene": gene, "organism": org, "error": str(e)[:120]}
        records.append(rec)
        n_new += 1
        print(gene, org, rec.get("accession"), rec.get("flag_counts"), rec.get("error", ""), flush=True)
        time.sleep(0.34)
    ok = [r for r in records if "accession" in r]
    json.dump({"n_accessions": len(ok), "records": records},
              open(args.out, "w"), indent=2)
    print("total accessions scored:", len(ok))


if __name__ == "__main__":
    main()
