"""Tests for the strict-arm power gate of certify_S_sumstat.

Invariants checked:
  1. Legacy strict (legacy=True) is unchanged on a 400-case random grid.
  2. Gated strict only ever turns legacy SURVIVES into UNVERIFIABLE (never alters REJECTED, never creates SURVIVES).
  3. Targeted verdicts and all targeted detail keys are byte-identical between legacy=False and legacy=True, and
     invariant to b_ref.
  4. When b_ref is omitted, the reference used by default is the breakeven slope b*.
  5. A well-powered arm with a null slope still certifies.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from proofcarry.sumstat import certify_S_sumstat  # noqa: E402
FAILED = []
TARGETED_KEYS = ("bias", "bias_se", "corrected", "z_point", "z_full", "claim_threshold", "can_still_pass",
                 "breakeven_b", "bias_fraction")


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond: FAILED.append(msg)


def controls(n, b_true, se_y, seed, x_mean=0.18, x_sd=0.05, se_x=0.03):
    rng = np.random.default_rng(seed)
    x = np.abs(rng.normal(x_mean, x_sd, n))
    y = b_true * x + rng.normal(0, se_y, n)
    return y, np.full(n, se_y), x, np.full(n, se_x)


def test_grid():
    rng = np.random.default_rng(20261018)
    trans, n_leg_ok, n_tgt_ok, n_cases = {}, 0, 0, 0
    for i in range(400):
        w = controls(int(rng.integers(8, 120)), float(rng.normal(0, 0.15)), float(rng.uniform(0.01, 0.12)), 1000 + i)
        beta = float(rng.uniform(0.05, 0.6)); se = float(rng.uniform(0.02, 0.12))
        t = (beta, se, float(rng.uniform(-0.9, 0.9)), float(rng.uniform(0.01, 0.08)))
        cp = float(rng.choice([5e-8, 5.7e-4, 0.05]))
        tg, sg, dg = certify_S_sumstat(*w, *t, claim_p=cp, legacy=False)
        tl, sl, dl = certify_S_sumstat(*w, *t, claim_p=cp, legacy=True)
        if sl.verdict == "NO CLAIM":
            continue
        n_cases += 1
        p = dl["slope"]["p"]
        n_leg_ok += sl.verdict == ("REJECTED" if p <= 0.05 else "SURVIVES")
        trans[(sl.verdict, sg.verdict)] = trans.get((sl.verdict, sg.verdict), 0) + 1
        tg2, _, dg2 = certify_S_sumstat(*w, *t, claim_p=cp, legacy=False, b_ref=1e-6)
        n_tgt_ok += (tg.verdict == tg2.verdict and all(dg[k] == dg2[k] for k in TARGETED_KEYS)
                     and all(dg[k] == dl[k] for k in TARGETED_KEYS))
    print("transitions legacy->gated:", trans, flush=True)
    check(n_cases > 200, f"grid has enough claim-present cases ({n_cases})")
    check(n_leg_ok == n_cases, "legacy strict == ungated rule on every case")
    allowed = {("REJECTED", "REJECTED"), ("SURVIVES", "SURVIVES"), ("SURVIVES", "UNVERIFIABLE")}
    check(set(trans) <= allowed, "only SURVIVES -> UNVERIFIABLE transitions occur")
    check(trans.get(("SURVIVES", "UNVERIFIABLE"), 0) > 0, "the gate fires on some cases")
    check(n_tgt_ok == n_cases, "targeted verdict and details unaffected by the gate and by b_ref")


def test_b_ref_and_powered():
    w = controls(60, 0.0, 0.01, 7)
    t = (0.5, 0.05, 0.1, 0.02)                      # z = 10, a* small -> b* large
    tg, sg, d = certify_S_sumstat(*w, *t, legacy=False)
    check(d["strict_b_ref_source"] == "breakeven" and d["strict_b_ref"] == d["breakeven_b"], "default b_ref = b*")
    if d["slope"]["p"] > 0.05:
        check(sg.verdict == "SURVIVES" and d["strict_powered"], "well-powered null arm SURVIVES")
        _, s2, d2 = certify_S_sumstat(*w, *t, legacy=False, b_ref=d["strict_mde_80"] / 2)
        check(s2.verdict == "UNVERIFIABLE" and d2["strict_b_ref_source"] == "caller", "caller b_ref below MDE -> UNVERIFIABLE")
    else:
        check(False, "seed 7 null slope unexpectedly significant; pick another seed")


if __name__ == "__main__":
    test_grid(); test_b_ref_and_powered()
    print(f"\n{len(FAILED)} failed" if FAILED else "\nALL PASS", flush=True)
    sys.exit(1 if FAILED else 0)
