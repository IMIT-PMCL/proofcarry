"""Tests for the symmetric targeted REJECTED branch of certify_S_sumstat.


The property that makes the change safe for already-recorded verdicts:

    symmetric REJECTED  =>  legacy REJECTED

because not can_still_pass implies |corr| + 1.96*seB < T, hence |corr| < T, hence |z_point| < z_claim. The SURVIVES
branch is evaluated first and is identical in both rules. So the change can only turn a legacy REJECTED into
UNVERIFIABLE; it can never alter a SURVIVES or turn an UNVERIFIABLE into a REJECTED.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from proofcarry.sumstat import certify_S_sumstat  # noqa: E402

FAILED = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond: FAILED.append(msg)


def controls(n, b_true, se_y, seed, x_mean=0.18, x_sd=0.05, se_x=0.03):
    rng = np.random.default_rng(seed)
    x = np.abs(rng.normal(x_mean, x_sd, n))
    y = b_true * x + rng.normal(0, se_y, n)
    return y, np.full(n, se_y), x, np.full(n, se_x)


def both(w, t, **kw):
    """Returns (symmetric targeted, legacy targeted, details) for the same inputs."""
    sym, _, det = certify_S_sumstat(*w, *t, legacy=False, **kw)
    leg, _, _ = certify_S_sumstat(*w, *t, legacy=True, **kw)
    return sym.verdict, leg.verdict, det


# ---- 1. Near-threshold claim with wide bias interval (PD mortality GBA1 p.E326K benchmark regime).
# Deterministic, not sampled: 19 controls on a fixed grid with y exactly b*x, and se_y chosen so that SE(b) lands near
# the 0.0755 of the real run. A sampled slope at this noise level is dominated by its own error (the first attempt
# realised -0.004 against a planted 0.07) and would not reliably exercise the branch under test.
_x = np.linspace(0.10, 0.30, 19)
w = (0.0698 * _x, np.full(19, 0.062), _x, np.full(19, 0.03))
t = (0.395, 0.1107, 0.7508, 0.0659)                       # Tan beta/SE, Nalls a*/SE for rs35749011
sym, leg, det = both(w, t, claim_p=0.05 / 88)
print(f"  near-threshold: b={det['slope']['b']:.4f}({det['slope']['se']:.4f}) corr={det['corrected']:.4f} "
      f"T={det['claim_threshold']:.4f} best={det['corrected_best']:.4f} z_point={det['z_point']:.3f} "
      f"breakeven_b={det['breakeven_b']:.4f}")
check(abs(det["z_point"]) < det["z_claim"], "near-threshold: the point-corrected effect misses the threshold")
check(det["can_still_pass"], "near-threshold: the favourable end of the bias interval still clears the threshold")
check(leg == "REJECTED", "near-threshold: legacy rule returns REJECTED")
check(sym == "UNVERIFIABLE", "near-threshold: symmetric rule returns UNVERIFIABLE")
check(0.01 < det["breakeven_b"] < 0.03, "near-threshold: breakeven_b is reported and ~0.018")

# ---- 2. Decisive bias: the whole interval is below the threshold -> REJECTED under both.
w2 = controls(60, 1.0, 0.01, seed=12)
sym2, leg2, det2 = both(w2, (0.30, 0.04, 0.25, 0.01))
check(not det2["can_still_pass"], "decisive: even the favourable end fails")
check(sym2 == leg2 == "REJECTED", "decisive: both rules REJECTED")

# ---- 3. No bias: SURVIVES under both, and the SURVIVES branch is byte-identical.
w3 = controls(60, 0.0, 0.01, seed=13)
sym3, leg3, det3 = both(w3, (0.30, 0.04, 0.02, 0.01))
check(sym3 == leg3 == "SURVIVES", "no-bias: both rules SURVIVES")

# ---- 4. Claim absent -> NO CLAIM under both, unchanged.
sym4, leg4, _ = both(w3, (0.02, 0.04, 0.02, 0.01))
check(sym4 == leg4 == "NO CLAIM", "absent claim: both rules NO CLAIM")

# ---- 5. The safety invariant, over a randomised grid: symmetric REJECTED => legacy REJECTED,
#         SURVIVES never changes, and UNVERIFIABLE never hardens into REJECTED.
rng = np.random.default_rng(99)
seen = {"softened": 0, "same": 0}
for k in range(400):
    n = int(rng.integers(8, 80)); b_true = float(rng.uniform(-0.6, 0.6)); se_y = float(rng.uniform(0.005, 0.08))
    wk = controls(n, b_true, se_y, seed=1000 + k)
    beta = float(rng.uniform(0.05, 0.6)) * rng.choice([-1, 1])
    tk = (beta, float(rng.uniform(0.01, 0.15)), float(rng.uniform(-0.9, 0.9)), float(rng.uniform(0.01, 0.1)))
    cp = float(rng.choice([5e-8, 0.05 / 88, 1e-5]))
    s, l, _ = both(wk, tk, claim_p=cp)
    if s == l: seen["same"] += 1
    elif l == "REJECTED" and s == "UNVERIFIABLE": seen["softened"] += 1
    else: check(False, f"invariant violated: legacy={l} symmetric={s} (case {k})")
    if l == "SURVIVES": check(s == "SURVIVES", f"SURVIVES changed in case {k}")
    if s == "REJECTED": check(l == "REJECTED", f"symmetric REJECTED without legacy REJECTED in case {k}")
print(f"  invariant grid: {seen['same']} identical, {seen['softened']} REJECTED->UNVERIFIABLE, 0 other transitions")
check(seen["softened"] > 0, "invariant grid: the softening transition is actually exercised")

print(("ALL PASS" if not FAILED else f"{len(FAILED)} FAILURES: " + "; ".join(FAILED)), flush=True)
sys.exit(0 if not FAILED else 1)
