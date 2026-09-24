import os
import subprocess
import sys

import numpy as np

from tools_outcome.loader import aggregate_events, load_targets

FIXTURE_CSV = """,Category,Count,Genotype Position,Indel with Mismatches,Inserted Bases,Length,Microhomology-Based,_Experiment
0,wildtype,1000.0,,,,,,0
1,del,100.0,1.0,no,,1.0,yes,0
2,del,50.0,2.0,no,,3.0,no,0
3,ins,80.0,0.0,no,A,1.0,na,0
4,wildtype,900.0,,,,,,1
5,del,200.0,1.0,no,,6.0,no,1
"""


def _write_fixture(tmp_path):
    p = tmp_path / "events.csv"
    p.write_text(FIXTURE_CSV)
    return str(p)


def test_aggregate_fixture(tmp_path):
    csv_path = _write_fixture(tmp_path)
    targets = ["A" * 55, "C" * 55]
    sites = aggregate_events(csv_path, targets)
    assert len(sites) == 2
    s0, s1 = sites
    # site 0: edited = 230; frameshift reads = del len1 (100) + ins len1 (80) = 180
    assert abs(s0.frameshift_frac - 180 / 230) < 1e-6
    assert abs(s0.mh_del_frac - 100 / 150) < 1e-6
    # site 1: del len 6 -> in-frame, frameshift 0
    assert s1.frameshift_frac == 0.0
    assert s1.mh_del_frac == 0.0


def test_load_targets_filters(tmp_path):
    p = tmp_path / "tg.txt"
    p.write_text("ACGTACGT\n\nNNNN\nTTTTGGGG\n")
    tg = load_targets(str(p))
    assert tg == ["ACGTACGT", "TTTTGGGG"]


def test_low_coverage_excluded(tmp_path):
    p = tmp_path / "events.csv"
    p.write_text(",Category,Count,Genotype Position,Indel with Mismatches,Inserted Bases,Length,Microhomology-Based,_Experiment\n"
                 "0,wildtype,10.0,,,,,,0\n1,del,5.0,1.0,no,,1.0,yes,0\n")
    assert aggregate_events(str(p), ["A" * 55]) == []
