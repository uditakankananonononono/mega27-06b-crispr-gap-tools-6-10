"""Tool 9 deployment walkthrough: real runs producing the paper's artifacts.

Part 1 (real, this machine): invoke the production CLI bioscreen path without
a commec install -> the wrapper must fail closed (BackendUnavailable), which
is the required behavior when screening cannot run.
Part 2 (format demonstration, labeled): the orchestration + hash-chained
audit on the deterministic fixture backend used by the test-suite, showing
the exact record format a production commec run emits. The production engine
(IBBIS commec + multi-GB databases) is environment-provided and exceeds this
workspace's free-tier footprint; the chain machinery itself is real code,
tested in tests/test_biosecurity.py.
Part 3 (real): tamper-evidence checks on the Part 2 chain - edit, delete,
reorder must all fail verification.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools_biosecurity.audit import verify_chain  # noqa: E402
from tools_biosecurity.screener import ScreenOutcome, screen_and_design  # noqa: E402

OUT = Path("results/bioscreen_audit_demo.json")

# Candidate oligos: real guide spacers from the repo's crisprSQL work plus
# adapter-like sequences, as a provider would receive them in an order batch.
CANDIDATES = {
    "order1_guideEMX1": "GAGTCCGAGCAGAAGAAGAA",
    "order1_guideVEGFA": "GGGTGGGGGGAGTTTGCTCC",
    "order1_fwdPrimer": "ACACTCTTTCCCTACACGAC",
    "order1_revPrimer": "GTGACTGGAGTTCAGACGTG",
}
PLANTED_MOTIF = "GGGGGG"  # fixture stands in for a regulated-hit pattern


class FixtureBackend:
    """Deterministic test-double backend (same class as the test suite)."""
    commec_bin = "fixture-no-commec-db"

    def screen(self, seqs):
        return [
            ScreenOutcome(sid, "FLAG" if PLANTED_MOTIF in s else "CLEAR",
                          hits=[{"motif": PLANTED_MOTIF}] if PLANTED_MOTIF in s else [],
                          engine="fixture")
            for sid, s in seqs.items()
        ]


def main() -> None:
    # Part 1: real fail-closed invocation of the production CLI.
    proc = subprocess.run(
        [sys.executable, "cli.py", "bioscreen",
         "g1=" + CANDIDATES["order1_guideEMX1"]],
        capture_output=True, text=True)
    part1 = {
        "command": "python cli.py bioscreen g1=<EMX1 spacer>",
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "",
        "behavior": "fail-closed: BackendUnavailable raised, no oligo accepted",
    }
    assert proc.returncode != 0 and "BackendUnavailable" in proc.stderr

    # Part 2: orchestration + chained audit (fixture backend, labeled).
    res = screen_and_design(CANDIDATES, FixtureBackend())
    part2 = {
        "purpose": ("audit-record format demonstration; production engine is "
                    "IBBIS commec (multi-GB databases, environment-provided)"),
        "engine": res["engine"],
        "accepted": res["accepted"],
        "blocked": res["blocked"],
        "chain_head": res["chain_head"],
        "chain": res["chain"],
    }

    # Part 3: real tamper-evidence checks.
    edited = [dict(r) for r in res["chain"]]
    edited[0]["verdict"] = "FLAG"  # g1 is CLEAR; flip it
    part3 = {
        "verify_intact": verify_chain(res["chain"]),
        "verify_after_verdict_edit": verify_chain(edited),
        "verify_after_line_deletion": verify_chain(res["chain"][1:]),
        "verify_after_reorder": verify_chain(res["chain"][::-1]),
    }
    assert part3 == {"verify_intact": True, "verify_after_verdict_edit": False,
                     "verify_after_line_deletion": False,
                     "verify_after_reorder": False}

    doc = {"part1_fail_closed_real_run": part1,
           "part2_audit_format_demo": part2,
           "part3_tamper_evidence_real": part3}
    OUT.write_text(json.dumps(doc, indent=2))
    print(json.dumps({"part1_returncode": part1["returncode"],
                      "part1_stderr_tail": part1["stderr_tail"],
                      "accepted": part2["accepted"], "blocked": part2["blocked"],
                      "chain_head": part2["chain_head"][:16] + "...",
                      **part3}, indent=2))


if __name__ == "__main__":
    main()
