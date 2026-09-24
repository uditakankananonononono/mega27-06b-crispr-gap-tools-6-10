"""MEGA27 item 6b: unified CLI for the five CRISPR gap tools (gaps 6-10).

Usage:
  python cli.py multiplex SPACER1 SPACER2 ... [--cutpos 100,205,900 --chrom chr1]
  python cli.py outcome CONTEXT55          # repair-outcome annotation (needs trained model - see scripts/)
  python cli.py riskflag CONTEXT CUT_POS
  python cli.py transfer A.fasta B.fasta   # fast kmer-ridge transfer check
  python cli.py bioscreen OligoID=SEQ ...  # requires commec install
"""
from __future__ import annotations

import argparse
import json

from tools_multiplex import screen_guides
from tools_riskflag import flag_cutsite


def _multiplex(args):
    cut = [int(x) for x in args.cutpos.split(",")] if args.cutpos else None
    res = screen_guides(args.spacers, cut_pos=cut, chrom=args.chrom)
    print(json.dumps({
        "compatibility": round(res.compatibility, 3),
        "per_guide": res.per_guide,
        "flagged_pairs": [
            {"a": p.guide_a, "b": p.guide_b, "duplex_mfe": round(p.heteroduplex_mfe, 2),
             "pam_competition": round(p.pam_competition, 3),
             "duplex_flag": p.duplex_flag, "pam_flag": p.pam_flag}
            for p in res.pair_reports if p.duplex_flag or p.pam_flag],
    }, indent=2))


def _riskflag(args):
    rep = flag_cutsite(args.context, args.cut_pos)
    print(json.dumps(rep.__dict__, indent=2))


def _transfer(args):
    from Bio import SeqIO
    from tools_transfer.evaluator import quick_transfer_eval
    def load(p):
        return [str(r.seq).upper()[:30] for r in SeqIO.parse(p, "fasta")]
    a, b = load(args.fasta_a), load(args.fasta_b)
    from crisprlib.featurize import gc_content
    v = quick_transfer_eval(a, [gc_content(s) for s in a],
                            b, [gc_content(s) for s in b], "A", "B")
    print(json.dumps(v.__dict__, indent=2, default=str))


def _bioscreen(args):
    from tools_biosecurity import CommecBackend, screen_and_design
    cands = dict(kv.split("=", 1) for kv in args.oligos)
    res = screen_and_design(cands, CommecBackend(db_dir=args.commec_db))
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("multiplex"); p.add_argument("spacers", nargs="+")
    p.add_argument("--cutpos"); p.add_argument("--chrom"); p.set_defaults(f=_multiplex)
    p = sub.add_parser("riskflag"); p.add_argument("context"); p.add_argument("cut_pos", type=int)
    p.set_defaults(f=_riskflag)
    p = sub.add_parser("transfer"); p.add_argument("fasta_a"); p.add_argument("fasta_b")
    p.set_defaults(f=_transfer)
    p = sub.add_parser("bioscreen"); p.add_argument("oligos", nargs="+")
    p.add_argument("--commec-db"); p.set_defaults(f=_bioscreen)
    args = ap.parse_args()
    args.f(args)


if __name__ == "__main__":
    main()
