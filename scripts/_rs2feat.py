"""Shared Rule Set 2 feature builder (extracted from beat_azimuth_v6)."""
from crisprlib.featurize import thermo_dg
DNA = "ACGT"
def rs2_features(ctx, aa, pct):
    ctx = ctx.upper(); guide = ctx[4:24]; feats = []
    for ch in ctx:
        feats.extend(1.0 if ch == b else 0.0 for b in DNA)
    for i in range(len(ctx) - 1):
        di = ctx[i:i+2]
        feats.extend(1.0 if di == a + b else 0.0 for a in DNA for b in DNA)
    for a in DNA:
        for b in DNA:
            feats.append(sum(1 for i in range(len(guide) - 1)
                             if guide[i:i+2] == a + b) / 19.0)
    feats.append(float(guide.count("G") + guide.count("C")))
    feats.append(thermo_dg(ctx)); feats.append(thermo_dg(guide))
    feats.append(thermo_dg(guide[:5])); feats.append(thermo_dg(guide[5:10]))
    feats.append(thermo_dg(guide[10:15])); feats.append(thermo_dg(guide[15:]))
    x_base = ctx[26] if len(ctx) > 26 else "N"
    for i in range(18, 28):
        di = ctx[i:i+2]
        for a in DNA:
            for c in DNA:
                for b in DNA:
                    feats.append(1.0 if (di == a + c and x_base == b) else 0.0)
    feats.append(aa / 100.0); feats.append(pct / 100.0)
    return feats

