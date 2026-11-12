"""Class-matched certificate engine.

Each confound class is closed by the primitive that consumes its matched auxiliary variable:
  S (selection-collider) -> negative-control-outcome (NCO) test: SURVIVES iff p_y<=alpha AND p_yneg>alpha
  B (batch)              -> within-batch stratified permutation: SURVIVES iff p_y<=alpha AND p_adj<=alpha
  U (pseudoreplication)  -> donor/unit-level permutation (the base severe test)

Statistical guardrail (Supplementary Information §S2): conditioning on Y is done by WITHIN-Y-STRATUM
centering (exact for categorical Y), NOT linear partialling — linear partialling is invalid under
selection because post-selection means are nonlinear in Y (Mills-ratio nonlinearity).

The PROOF-CARRYING-DATA gate: an S-certificate is REFUSED (returns UNVERIFIABLE) unless a co-selected
NCO is available — verification of the collider class is *undefined* without the auxiliary variable.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np

ALPHA = 0.05


@dataclass
class Certificate:
    verdict: str            # "SURVIVES" | "REJECTED" | "UNVERIFIABLE"
    confound_class: str
    p_y: Optional[float] = None
    p_aux: Optional[float] = None       # p_yneg (S) or p_adj (B)
    n_units: Optional[int] = None
    reason: str = ""

    def __repr__(self):
        return (f"Certificate({self.verdict} [{self.confound_class}] p_y={self.p_y} "
                f"p_aux={self.p_aux} n_units={self.n_units}: {self.reason})")


def _unit_composite(X, y, unit):
    """Donor/unit-level composite: mean of the (already-selected) claimed features per unit."""
    unit = np.asarray(unit)
    comp = X.mean(axis=1) if X.ndim == 2 else np.asarray(X, float)
    uids = np.unique(unit)
    cu = np.array([comp[unit == u].mean() for u in uids])
    yu = np.array([y[unit == u][0] for u in uids])
    return uids, cu, yu


def _perm_p(cu, yu, rng, nperm=1999):
    obs = abs(cu[yu == 1].mean() - cu[yu == 0].mean())
    c = 1
    for _ in range(nperm):
        yp = rng.permutation(yu)
        if abs(cu[yp == 1].mean() - cu[yp == 0].mean()) >= obs:
            c += 1
    return c / (nperm + 1)


def _within_y_center(v, y):
    """Exact conditioning on categorical Y (NOT linear partialling)."""
    out = np.array(v, float)
    for g in np.unique(y):
        m = y == g
        out[m] = out[m] - out[m].mean()
    return out


def certify_U(X, y, unit, alpha=ALPHA, seed=0):
    """Base severe test: donor-level permutation. SURVIVES iff p_y<=alpha."""
    rng = np.random.default_rng(seed)
    uids, cu, yu = _unit_composite(X, y, unit)
    if min((yu == 1).sum(), (yu == 0).sum()) < 2:
        return Certificate("UNVERIFIABLE", "U", n_units=len(uids), reason="fewer than 2 units per arm")
    p = _perm_p(cu, yu, rng)
    return Certificate("SURVIVES" if p <= alpha else "REJECTED", "U", p_y=round(float(p), 4),
                       n_units=len(uids), reason="donor-level permutation severe test")


Z_POWER = 2.486   # z_{0.95} + z_{0.80}: 80% power, one-sided alpha = 0.05


def rho_min(n: int) -> float:
    """Smallest within-Y NCO correlation detectable with 80% power at n units (one-sided alpha = 0.05)."""
    return Z_POWER / np.sqrt(n)


def certify_S(X, y, unit, w, co_selected: bool, alpha=ALPHA, seed=0, rho_ref=None, rho_negligible=None,
              legacy=False):
    """Selection-collider NCO test. REFUSES (UNVERIFIABLE) unless a CO-SELECTED W is supplied.
    SURVIVES iff the composite predicts the real outcome (p_y<=alpha) but NOT the co-selected
    negative control (p_yneg>alpha) — a collider artifact predicts both.

    Default (power-aware rule): a SURVIVES verdict requires a declared rho_ref.
    Without one the arm's power is unknown (an ungated rule can false-certify up to 67% of collider nulls at n = 600),
    so the verdict is UNVERIFIABLE ("no power reference declared"); REJECTED is unchanged.
    legacy=True reproduces the ungated rule without a minimum-detectable-effect check.

    Power-aware mode (rho_ref / rho_negligible; with both None and legacy=True, the legacy rule):
      rho_ref        pre-registered collider signature (within-Y |corr(X, W)|) the test must be able to detect.
                     If the NCO arm does not fire and rho_min(n) > rho_ref, the arm was under-powered for the
                     signature that matters -> UNVERIFIABLE, not SURVIVES.
      rho_negligible fixed effect-size floor: the NCO arm fires only if p_yneg <= alpha AND |r| >= rho_negligible
                     (high-n regime, where p <= alpha is reached by negligible couplings). Must be < rho_ref."""
    if rho_ref is not None and rho_negligible is not None and not rho_negligible < rho_ref:
        raise ValueError("rho_negligible must be below rho_ref, else a detectable signature could be called negligible")
    if w is None:
        return Certificate("UNVERIFIABLE", "S", reason="no negative-control outcome supplied")
    if not co_selected:
        return Certificate("UNVERIFIABLE", "S",
                           reason="NCO is not co-selected — an independent control cannot certify the "
                                  "collider class (power=alpha by construction; §S2 failure mode 1)")
    rng = np.random.default_rng(seed)
    uids, cu, yu = _unit_composite(X, y, unit)
    wu = np.array([np.asarray(w)[np.asarray(unit) == u][0] for u in uids], float)
    if min((yu == 1).sum(), (yu == 0).sum()) < 2:
        return Certificate("UNVERIFIABLE", "S", n_units=len(uids), reason="fewer than 2 units per arm")
    p_y = _perm_p(cu, yu, rng)
    # NCO association: within-Y-conditional correlation of composite and W (exact conditioning)
    rc, rw = _within_y_center(cu, yu), _within_y_center(wu, yu)
    r = 0.0
    if rc.std() < 1e-9 or rw.std() < 1e-9:
        p_yneg = 1.0
    else:
        r = abs(np.corrcoef(rc, rw)[0, 1])
        from scipy import stats
        p_yneg = float(2 * (1 - stats.norm.cdf(r * np.sqrt(len(uids)))))
    if rho_ref is None and rho_negligible is None:              # legacy rule
        survives = (p_y <= alpha) and (p_yneg > alpha)
        if survives and not legacy:
            return Certificate("UNVERIFIABLE", "S", p_y=round(float(p_y), 4), p_aux=round(float(p_yneg), 4),
                               n_units=len(uids), reason=f"no power reference declared: the control did not fire, but "
                                                         f"rho_min(n)={rho_min(len(uids)):.4f} is unchecked (pass rho_ref, "
                                                         f"or legacy=True for the ungated rule)")
        return Certificate("SURVIVES" if survives else "REJECTED", "S",
                           p_y=round(float(p_y), 4), p_aux=round(float(p_yneg), 4), n_units=len(uids),
                           reason="NCO test: SURVIVES iff p_y<=a AND p_yneg>a (co-selected control does not fire)")
    fires = (p_yneg <= alpha) and (rho_negligible is None or r >= rho_negligible)
    floor = rho_min(len(uids))
    tag = f"|r|={r:.4f}, rho_min(n)={floor:.4f}, rho_ref={rho_ref}, rho_negligible={rho_negligible}"
    if p_y > alpha:
        v, why = "REJECTED", "no claim at alpha"
    elif fires:
        v, why = "REJECTED", "co-selected control fires"
    elif rho_ref is not None and floor > rho_ref:
        v, why = "UNVERIFIABLE", "NCO arm under-powered for the reference collider signature"
    else:
        v, why = "SURVIVES", "co-selected control does not fire and the arm was powered for rho_ref"
    return Certificate(v, "S", p_y=round(float(p_y), 4), p_aux=round(float(p_yneg), 4), n_units=len(uids),
                       reason=f"power-aware NCO test: {why} ({tag})")


def certify_B(X, y, unit, batch, alpha=ALPHA, seed=0):
    """Batch test: label-permutation WITHIN each observed batch stratum (holds the between-batch
    contribution in the null). SURVIVES iff p_y<=alpha AND p_adj<=alpha."""
    if batch is None:
        return Certificate("UNVERIFIABLE", "B", reason="no batch indicator supplied")
    rng = np.random.default_rng(seed)
    uids, cu, yu = _unit_composite(X, y, unit)
    bu = np.array([np.asarray(batch)[np.asarray(unit) == u][0] for u in uids])
    if min((yu == 1).sum(), (yu == 0).sum()) < 2:
        return Certificate("UNVERIFIABLE", "B", n_units=len(uids), reason="fewer than 2 units per arm")
    p_y = _perm_p(cu, yu, rng)
    obs = abs(cu[yu == 1].mean() - cu[yu == 0].mean()); c = 1; nperm = 1999
    for _ in range(nperm):
        yp = yu.copy()
        for b in np.unique(bu):                     # permute labels within each batch stratum
            idx = np.where(bu == b)[0]
            yp[idx] = rng.permutation(yu[idx])
        if abs(cu[yp == 1].mean() - cu[yp == 0].mean()) >= obs:
            c += 1
    p_adj = c / (nperm + 1)
    survives = (p_y <= alpha) and (p_adj <= alpha)
    return Certificate("SURVIVES" if survives else "REJECTED", "B",
                       p_y=round(float(p_y), 4), p_aux=round(float(p_adj), 4), n_units=len(uids),
                       reason="within-batch stratified permutation")


def certify(X, y, unit, confound_class="U", w=None, co_selected=True, batch=None, alpha=ALPHA, seed=0,
            rho_ref=None, rho_negligible=None, legacy=False):
    """Dispatch to the class-matched primitive. X = selected claimed-feature matrix (cells x feats)
    or a per-cell composite; y = binary label per cell; unit = unit id per cell.
    rho_ref / rho_negligible / legacy: the S rule (see certify_S; SURVIVES needs rho_ref unless legacy=True)."""
    X = np.asarray(X, float); y = np.asarray(y).astype(int)
    c = confound_class.upper()
    if c == "S":
        return certify_S(X, y, unit, w, co_selected, alpha, seed, rho_ref=rho_ref, rho_negligible=rho_negligible,
                         legacy=legacy)
    if c == "B":
        return certify_B(X, y, unit, batch, alpha, seed)
    if c in ("U", "A", "M", "C"):
        return certify_U(X, y, unit, alpha, seed)
    return Certificate("UNVERIFIABLE", c, reason=f"unknown confound class {confound_class!r}")
