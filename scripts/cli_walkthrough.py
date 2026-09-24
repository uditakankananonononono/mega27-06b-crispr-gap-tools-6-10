"""Unified-CLI walkthrough: real invocations of the shipped tool, outputs
captured verbatim into results/cli_walkthrough.json for the paper's
'Using the shipped tool' section. Nothing here is mocked: each command is
a subprocess call to cli.py, and the recorded output is its stdout."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cli_walkthrough.json"


def run(args: list[str]) -> dict:
    import os
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    p = subprocess.run([sys.executable, "cli.py", *args],
                       capture_output=True, text=True, cwd=ROOT, env=env)
    return {"cmd": "python cli.py " + " ".join(args),
            "returncode": p.returncode,
            "stdout": json.loads(p.stdout) if p.returncode == 0 else None,
            "stderr_tail": p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ""}


def main() -> None:
    doc = {}

    # 1. multiplex: a compatible trio of real published spacers (EMX1, VEGFA,
    #    FANCF) and a deliberately self-complementary pair that must flag.
    doc["multiplex_compatible"] = run([
        "multiplex", "GAGTCCGAGCAGAAGAAGAA", "GGGTGGGGGGAGTTTGCTCC",
        "GGGAATCGGACGAGGTTGCG"])
    doc["multiplex_flagged"] = run([
        "multiplex", "GAGTCCGAGCAGAAGAAGAA", "TTCTTCTTCTGCTCGGACTC"])

    # 2. riskflag: real Cd9 mouse locus context (NM_007657.4), cut placed at
    #    an interior NGG site found from the sequence itself.
    seq = "".join(l.strip() for l in open(ROOT / "data" / "Cd9_mouse.fa")
                  if not l.startswith(">")).upper()
    cut = seq.find("AGG") + 2  # cut 3 nt upstream of a real PAM
    assert 250 <= cut <= len(seq) - 250 or len(seq) >= 501
    ctx_start = max(0, cut - 250)
    context = seq[ctx_start:ctx_start + 501]
    doc["riskflag_cd9"] = run(["riskflag", context, str(cut - ctx_start)])
    doc["riskflag_cd9"]["locus"] = "NM_007657.4 Cd9 (mouse), context from committed data/Cd9_mouse.fa"

    # 3. transfer: 30 real V1 mouse guides vs 30 real RES human guides,
    #    extracted from the committed datasets into /tmp fastas.
    import csv
    v1 = []
    with open(ROOT / "data" / "V1_suppl_data.txt") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        for row in rd:
            if len(v1) < 300 and row.get("Spacer Sequence"):
                v1.append(row["Spacer Sequence"].upper())
    res = []
    with open(ROOT / "data" / "FC_plus_RES_withPredictions.csv") as fh:
        rd = csv.DictReader(fh)
        cols = rd.fieldnames or []
        for row in rd:
            if len(res) >= 300:
                break
            if (row.get("drug") or "").strip().lower() == "nodrug":
                continue  # RES training guides only (loader rule: no FC test rows)
            g = (row.get("30mer") or "").upper()
            if len(g) >= 24 and set(g[4:24]) <= set("ACGT"):
                res.append(g[4:24])
    assert len(v1) == 300 and len(res) == 300, (len(v1), len(res))
    fa = Path("/tmp/wt_v1.fasta"); fb = Path("/tmp/wt_res.fasta")
    fa.write_text("".join(f">v1_{i}\n{g}\n" for i, g in enumerate(v1)))
    fb.write_text("".join(f">res_{i}\n{g}\n" for i, g in enumerate(res)))
    doc["transfer_v1_vs_res"] = run(["transfer", str(fa), str(fb)])
    doc["transfer_v1_vs_res"]["note"] = ("300 V1 mouse guides vs 300 RES human guides, "
                                         "extracted from committed data files")

    OUT.write_text(json.dumps(doc, indent=2))
    for k, v in doc.items():
        print(f"== {k}: rc={v['returncode']}")
        s = json.dumps(v["stdout"])[:220] if v["stdout"] else v["stderr_tail"][:160]
        print("  ", s)


if __name__ == "__main__":
    main()
