"""Tests for the opt-in power-aware S rule (certify_S rho_ref / rho_negligible), 2026-10-16.
"""
import subprocess, sys, types
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from proofcarry.certificate import certify_S, certify, rho_min  # noqa: E402


def legacy_module():
    """certify_S exactly as committed before the power branch (git 7bc6d88), for equivalence."""
    src = subprocess.run(["git", "show", "7bc6d88:tools/proofcarry/proofcarry/certificate.py"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType("legacy_cert"); sys.modules["legacy_cert"] = mod   # dataclasses look the module up
    exec(compile(src, "legacy_cert", "exec"), mod.__dict__)
    return mod


def collider(n, strength, seed):
    """y binary label, X composite, W co-selected; selection on X + W + y-dependent term induces within-y corr(X, W)."""
    rng = np.random.default_rng(seed)
    N = n * 6
    y = rng.integers(0, 2, N); W = rng.normal(size=N); X = rng.normal(size=N) + 0.3 * y
    keep = (strength * (X + W) + rng.normal(size=N)) > np.quantile(strength * (X + W) + rng.normal(size=N), 1 - 1 / 6)
    idx = np.where(keep)[0][:n]
    return X[idx], y[idx], W[idx]


def test_legacy_equivalence():
    L = legacy_module()
    for s in range(12):
        X, y, W = collider(400, [0.0, 0.3, 1.0][s % 3], s)
        a = certify_S(X, y, np.arange(len(X)), W, True, seed=s, legacy=True)
        b = L.certify_S(X, y, np.arange(len(X)), W, True, seed=s)
        assert (a.verdict, a.p_y, a.p_aux, a.reason) == (b.verdict, b.p_y, b.p_aux, b.reason), (a, b)


def test_rho_min():
    assert abs(rho_min(6743) - 0.03027) < 1e-4 and abs(rho_min(60268) - 0.010127) < 1e-5


def test_underpowered_is_unverifiable():
    # weak signature, small n: legacy SURVIVES (miss) but power-aware -> UNVERIFIABLE when rho_min(n) > rho_ref
    rng = np.random.default_rng(1); n = 600
    y = rng.integers(0, 2, n); W = rng.normal(size=n); X = 0.5 * y + 0.02 * W + rng.normal(size=n)
    leg = certify_S(X, y, np.arange(n), W, True, legacy=True)
    pa = certify_S(X, y, np.arange(n), W, True, rho_ref=0.02)
    assert leg.verdict == "SURVIVES", leg
    assert rho_min(n) > 0.02 and pa.verdict == "UNVERIFIABLE", pa
    # same data, a reference the arm can detect (rho_ref above rho_min(n)) -> SURVIVES
    pa2 = certify_S(X, y, np.arange(n), W, True, rho_ref=0.15)
    assert pa2.verdict == "SURVIVES", pa2


def test_fires_still_rejects_and_negligible_floor():
    X, y, W = collider(800, 1.0, 3)
    assert certify_S(X, y, np.arange(len(X)), W, True, rho_ref=0.05).verdict == "REJECTED"
    # tiny real coupling at large n: p <= alpha but |r| below the negligibility floor -> not a firing
    rng = np.random.default_rng(5); n = 3000
    y = rng.integers(0, 2, n); W = rng.normal(size=n); X = 0.4 * y + 0.06 * W + rng.normal(size=n)
    leg = certify_S(X, y, np.arange(n), W, True, legacy=True)
    pa = certify_S(X, y, np.arange(n), W, True, rho_ref=0.12, rho_negligible=0.10)
    assert leg.verdict == "REJECTED" and pa.verdict == "SURVIVES", (leg, pa)


def test_floor_ordering_and_dispatcher():
    X, y, W = collider(300, 0.3, 9)
    try:
        certify_S(X, y, np.arange(len(X)), W, True, rho_ref=0.05, rho_negligible=0.05); raise AssertionError("no error")
    except ValueError:
        pass
    a = certify(X, y, np.arange(len(X)), "S", w=W, rho_ref=0.2)
    assert "power-aware" in a.reason, a



def test_default_needs_power_reference():
    # 2026-10-17 default: without rho_ref a non-firing control gives UNVERIFIABLE, never SURVIVES; REJECTED unchanged
    rng = np.random.default_rng(1); n = 600
    y = rng.integers(0, 2, n); W = rng.normal(size=n); X = 0.5 * y + 0.02 * W + rng.normal(size=n)
    d = certify_S(X, y, np.arange(n), W, True)
    assert d.verdict == "UNVERIFIABLE" and "no power reference" in d.reason, d
    assert certify_S(X, y, np.arange(n), W, True, legacy=True).verdict == "SURVIVES"
    Xc, yc, Wc = collider(800, 1.0, 3)
    assert certify_S(Xc, yc, np.arange(len(Xc)), Wc, True).verdict == "REJECTED"
    assert certify(X, y, np.arange(n), "S", w=W).verdict == "UNVERIFIABLE"


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name)
