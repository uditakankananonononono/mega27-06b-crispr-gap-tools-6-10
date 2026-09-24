import pytest

from tools_biosecurity.screener import (BackendUnavailable, CommecBackend,
                                        ScreenOutcome, screen_and_design)


class FixtureBackend:
    """Deterministic test double implementing the backend interface: flags any
    sequence containing a planted 'regulated' motif, real logic otherwise."""
    commec_bin = "fixture"

    def __init__(self, motif="AAAA"):
        self.motif = motif

    def screen(self, seqs):
        return [ScreenOutcome(sid, "FLAG" if self.motif in s else "CLEAR",
                              engine="fixture") for sid, s in seqs.items()]


def test_commec_unavailable_raises():
    b = CommecBackend(db_dir=None, commec_bin=None)
    assert not b.available()
    with pytest.raises(BackendUnavailable):
        b.screen({"x": "ACGT"})


def test_screen_and_design_blocks_flagged():
    cands = {"g1": "ACGTACGTACGTACGTACGT", "g2": "TTAAAACCCCGGGGTTTTGG"}
    res = screen_and_design(cands, FixtureBackend())
    assert res["accepted"] == ["g1"]
    assert res["blocked"] == ["g2"]
    assert len(res["audit"]) == 2
    statuses = {a["id"]: a["status"] for a in res["audit"]}
    assert statuses == {"g1": "CLEAR", "g2": "FLAG"}


def test_all_clear():
    cands = {"g1": "ACGTACGTACGTACGTACGT", "g2": "GGGGCCCCACACGTGTACAC"}
    res = screen_and_design(cands, FixtureBackend(motif="ZZZZZ"))
    assert res["accepted"] == ["g1", "g2"]
    assert res["blocked"] == []


def test_audit_chain_intact_verifies():
    from tools_biosecurity.audit import GENESIS, verify_chain
    cands = {"g1": "ACGTACGTACGTACGTACGT", "g2": "TTAAAACCCCGGGGTTTTGG",
             "g3": "GGGGCCCCACACGTGTACAC"}
    res = screen_and_design(cands, FixtureBackend())
    assert len(res["chain"]) == 3
    assert res["chain"][0]["prev_chain"] == GENESIS
    assert res["chain"][-1]["chain"] == res["chain_head"]
    assert verify_chain(res["chain"])


def test_audit_chain_tamper_evidence():
    from tools_biosecurity.audit import verify_chain
    cands = {"g1": "ACGTACGTACGTACGTACGT", "g2": "TTAAAACCCCGGGGTTTTGG"}
    res = screen_and_design(cands, FixtureBackend())
    edited = [dict(r) for r in res["chain"]]
    edited[0]["verdict"] = "FLAG"          # edit a verdict
    assert not verify_chain(edited)
    assert not verify_chain(res["chain"][1:])   # delete a line
    assert not verify_chain(res["chain"][::-1])  # reorder


def test_audit_chain_hides_sequence():
    cands = {"g1": "ACGTACGTACGTACGTACGT"}
    res = screen_and_design(cands, FixtureBackend())
    blob = str(res["chain"])
    assert "ACGTACGTACGTACGTACGT" not in blob  # only SHA-256 is recorded
    assert len(res["chain"][0]["seq_sha256"]) == 64
