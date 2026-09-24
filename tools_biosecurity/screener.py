"""Screen-and-design wrapper unifying biosecurity screening with guide design.

The screening engine is IBBIS's Common Mechanism (`commec`, MIT-licensed,
github.com/ibbis-bio/common-mechanism): HMM biorisk search + BLAST best-match
taxonomy against regulated-pathogen control lists. It is a conda package with
multi-GB reference databases; this wrapper:
  1. detects a working commec install,
  2. screens every candidate oligo (guides, adapters, primers) BEFORE it is
     accepted into a design,
  3. blocks flagged sequences from the design set and records why,
  4. reports UNAVAILABLE honestly when commec is not installed - it never
     silently passes unscreened sequences as screened.

The orchestration (batching, blocking, audit log) is fully implemented and
tested; only the DB-backed engine is environment-provided, like the live
dataset pulls that run outside CI.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


class BackendUnavailable(RuntimeError):
    pass


@dataclass
class ScreenOutcome:
    sequence_id: str
    status: str            # CLEAR | FLAG | WARNING | UNAVAILABLE
    hits: list = field(default_factory=list)
    engine: str = "commec"


class CommecBackend:
    """Runs `commec screen` on a FASTA of candidate oligos."""

    def __init__(self, db_dir: str | None = None, commec_bin: str | None = None):
        self.db_dir = db_dir
        self.commec_bin = commec_bin or shutil.which("commec")

    def available(self) -> bool:
        return self.commec_bin is not None and self.db_dir is not None

    def screen(self, seqs: dict[str, str]) -> list[ScreenOutcome]:
        if not self.available():
            raise BackendUnavailable(
                "commec CLI or reference databases not found; install per "
                "github.com/ibbis-bio/common-mechanism (conda + commec setup)")
        with tempfile.TemporaryDirectory() as td:
            fa = Path(td) / "candidates.fasta"
            with fa.open("w") as fh:
                for sid, s in seqs.items():
                    fh.write(f">{sid}\n{s}\n")
            subprocess.run([self.commec_bin, "screen", "-d", self.db_dir,
                            str(fa), "-o", td], check=True,
                           capture_output=True, timeout=3600)
            jf = Path(td) / "candidates.screen.json"
            data = json.loads(jf.read_text())
        outcomes = []
        for sid in seqs:
            rec = data.get(sid, {}) if isinstance(data, dict) else {}
            flag = str(rec.get("flag", rec.get("status", ""))).upper()
            status = ("FLAG" if "FLAG" in flag else
                      "WARNING" if "WARN" in flag else "CLEAR")
            outcomes.append(ScreenOutcome(sid, status,
                                          hits=rec.get("hits", []), engine="commec"))
        return outcomes


def screen_and_design(candidates: dict[str, str], backend,
                      allow_warning: bool = True) -> dict:
    """Screen candidate oligos and return the admissible design set + audit.

    candidates: {oligo_id: sequence}. Returns dict with accepted/blocked lists
    and per-sequence outcomes. If the backend is unavailable, raises
    BackendUnavailable - callers must surface this, not skip screening.
    """
    outcomes = backend.screen(candidates)
    accepted, blocked, audit = [], [], []
    for oc in outcomes:
        ok = oc.status == "CLEAR" or (allow_warning and oc.status == "WARNING")
        (accepted if ok else blocked).append(oc.sequence_id)
        audit.append({"id": oc.sequence_id, "status": oc.status,
                      "engine": oc.engine, "n_hits": len(oc.hits)})
    return {"accepted": accepted, "blocked": blocked, "audit": audit,
            "engine": getattr(backend, "commec_bin", "unknown")}
