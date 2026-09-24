"""Hash-chained, append-only audit log for biosecurity screening runs.

Each record is (SHA256(sequence), backend version, verdict, flags, UTC time)
chained as c_i = SHA256(c_{i-1} || r_i) so deletion, insertion, reordering,
or edit of any record is detectable from the chain head alone. This is the
tamper-evidence property a synthesis-provider deployment needs: a screening
log that can be edited after the fact is worse than no log.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

GENESIS = "0" * 64


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build_chain(records: list[dict]) -> list[dict]:
    """Append chain hashes to raw screening records, in order.

    Each input record needs at least: sequence (str), verdict (str),
    engine (str). Optional: flags (list), n_hits (int). Returns new dicts
    with added keys seq_sha256, ts_utc, prev_chain, chain. Input order is
    the canonical order; verification detects any deviation from it.
    """
    out, prev = [], GENESIS
    for rec in records:
        row = {
            "seq_sha256": _sha256(rec["sequence"]),
            "engine": rec["engine"],
            "verdict": rec["verdict"],
            "flags": list(rec.get("flags", [])),
            "n_hits": int(rec.get("n_hits", len(rec.get("flags", [])))),
            "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "prev_chain": prev,
        }
        payload = json.dumps(row, sort_keys=True)
        row["chain"] = _sha256(prev + payload)
        prev = row["chain"]
        out.append(row)
    return out


def verify_chain(chain: list[dict]) -> bool:
    """Recompute the chain; True iff every link and the head are intact."""
    prev = GENESIS
    for row in chain:
        if row.get("prev_chain") != prev:
            return False
        body = {k: v for k, v in row.items() if k != "chain"}
        if _sha256(prev + json.dumps(body, sort_keys=True)) != row.get("chain"):
            return False
        prev = row["chain"]
    return True
