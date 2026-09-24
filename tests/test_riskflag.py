from tools_riskflag import flag_cutsite


def test_clean_sequence_low_risk():
    import random
    rng = random.Random(42)
    ctx = "".join(rng.choice("ACGT") for _ in range(600))
    rep = flag_cutsite(ctx, 300)
    assert rep.flag in ("LOW", "MODERATE")
    assert rep.n_repeat_pairs <= 5


def test_planted_direct_repeats_high_risk():
    import random
    rng = random.Random(0)
    def rand(n):
        return "".join(rng.choice("ACGT") for _ in range(n))
    repeat = "GATTACAGATTACAGATTACA"
    left = rand(200) + repeat + rand(50)
    right = rand(50) + repeat + rand(200)
    ctx = left + right
    rep = flag_cutsite(ctx, len(left))
    assert rep.n_repeat_pairs >= 1
    assert rep.max_deletion_span >= 100
    assert rep.score > 0.2


def test_homopolymer_detection():
    ctx = "A" * 50 + "ACGT" * 75 + "T" * 50
    rep = flag_cutsite(ctx, 200)
    assert rep.max_homopolymer >= 40


def test_asymmetry_metric():
    ctx = "ACGT" * 150
    rep = flag_cutsite(ctx, 300)
    assert 0.0 <= rep.asymmetry <= 1.0
