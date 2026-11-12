"""Selection-collider (S) certificate on GWAS summary statistics (case-only / index-event setting).

Setting: a claim that SNP g* affects a trait Y measured only inside a selected set S (e.g. disease severity
among cases). Selection on S induces, for every SNP j, a bias in the case-only effect that is linear in the
SNP's own effect on selection (Dudbridge et al., Nat Commun 2019):

    beta_sev_j = s_j + b * beta_inc_j + e_j,   b common to all SNPs (first order).

The co-selected negative controls W are the known causes of S (published susceptibility SNPs), assumed to have
no direct effect on Y, or effects uncorrelated with their selection effects (InSIDE). Two decision rules:

  strict   -- the certify_S analogue: the NCO arm "fires" if b != 0 at alpha. SURVIVES iff the claim is present
              and the NCO arm does not fire. Rejects the whole selected set, whatever g*'s own selection effect.
  targeted -- uses the linear structure: the bias on g* is B = b * beta_inc(g*). Three-valued and power-aware:
              SURVIVES      the corrected effect beta_sev(g*) - B stays past the claim threshold even after
                            propagating the uncertainty of B;
              REJECTED      even the most favourable bias inside its 95% interval leaves the claim below the
                            threshold (or confidently flips its sign);
              UNVERIFIABLE  the bias interval spans the claim threshold, so the data cannot decide either way.
              NO CLAIM      the claim is not present at its own threshold in these statistics.

Symmetry of the targeted rule: an asymmetric rule that returns REJECTED whenever the *point*-corrected
effect misses the threshold—however wide the bias interval is—while reserving UNVERIFIABLE for the passing side is
generous to uncertainty when the point estimate passes and harsh when it fails, mislabelling near-threshold claims.
The default symmetric rule returns REJECTED only when even the most favourable bias in the 95% confidence interval leaves
the claim below the threshold (or confidently flips its sign); otherwise, when the interval spans the threshold, it
returns UNVERIFIABLE. `legacy=True` reproduces the asymmetric rule.

Strict-arm power gate: an ungated strict SURVIVES means only that the negative-control slope test did not fire, however
imprecise the slope estimate is. The default strict reference is the claim's own breakeven slope
b* = (|beta_sev| - z_claim*se_sev)/|a*| (the slope at which selection bias would erase the claim; a caller may pass
`b_ref` to override). When the negative-control arm does not fire and MDE_80(SE(b)) = 2.802*SE(b) > b_ref, the strict
rule returns UNVERIFIABLE instead of SURVIVES. `legacy=True` reproduces the ungated strict rule.

Only aggregates enter here (per-SNP effect estimates and SEs), never individual rows.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
from scipy import stats

from .certificate import Certificate, ALPHA

GW = 5e-8
Z95 = 1.959963984540054          # the multiplier used for bias_ci and for both sides of the targeted rule


@dataclass
class Slope:
    b: float; se: float; b_raw: float; se_raw: float; se_model: float; se_sandwich: float
    reliability: float; tau2: float; n_snps: int; p: float

    def as_dict(self): return asdict(self)


def _z(p):  # two-sided p -> |z|
    p = np.clip(np.asarray(p, float), 1e-300, 1.0)
    return stats.norm.isf(p / 2)


def _p(z):  # |z| -> two-sided p
    return float(2 * stats.norm.sf(abs(float(z))))


def se_from_p(beta, p):
    """Back-derive an SE from an effect and a two-sided p (for files that report OR and P only)."""
    beta = np.abs(np.asarray(beta, float)); z = _z(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.where(z > 0, beta / z, np.nan)
    return se


def slope_through_origin(x, se_x, y, se_y, dilution=True, max_iter=50):
    """b in y = b*x + e, weights 1/(se_y^2 + tau2), tau2 by moments; Hedges-Olkin dilution correction for noisy x."""
    x, se_x, y, se_y = (np.asarray(v, float) for v in (x, se_x, y, se_y))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(se_x) & np.isfinite(se_y) & (se_y > 0)
    x, se_x, y, se_y = x[m], se_x[m], y[m], se_y[m]; n = len(x)
    if n < 3:
        return Slope(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, n, np.nan)
    tau2 = 0.0
    for _ in range(max_iter):                      # moment estimate of between-SNP variance (DL-style), iterated
        w = 1 / (se_y ** 2 + tau2)
        b = np.sum(w * x * y) / np.sum(w * x * x)
        r = y - b * x
        q = np.sum(w * r * r)                      # expected ~ (n-1) under tau2 correct
        new = max(0.0, tau2 + (q - (n - 1)) / np.sum(w))
        if abs(new - tau2) < 1e-12: tau2 = new; break
        tau2 = new
    w = 1 / (se_y ** 2 + tau2)
    sxx = np.sum(w * x * x)
    b_raw = np.sum(w * x * y) / sxx
    r = y - b_raw * x
    se_model = np.sqrt(1 / sxx)
    se_sand = np.sqrt(np.sum((w * x * r) ** 2)) / sxx * np.sqrt(n / max(n - 1, 1))
    se_raw = max(se_model, se_sand)
    rel = 1.0
    if dilution:                                   # E[sum w xhat^2] = sum w x_true^2 + sum w se_x^2
        rel = float(np.clip((sxx - np.sum(w * se_x ** 2)) / sxx, 0.2, 1.0))
    b, se = b_raw / rel, se_raw / rel
    return Slope(float(b), float(se), float(b_raw), float(se_raw), float(se_model), float(se_sand),
                 rel, float(tau2), int(n), _p(b / se))


def certify_S_sumstat(beta_sev_w, se_sev_w, beta_inc_w, se_inc_w,
                      beta_sev_t, se_sev_t, beta_inc_t, se_inc_t,
                      alpha=ALPHA, claim_p=GW, dilution=True, legacy=False, b_ref=None):
    """Returns (targeted Certificate, strict Certificate, details dict). Effects of W and the target must be
    oriented to the same allele convention in both studies (severity and selection effect per the same allele).
    legacy=True restores the pre-2026-10-17 asymmetric REJECTED branch and the ungated strict arm (module docstring).
    b_ref: slope the strict arm must be powered to detect; default None = the breakeven slope b*."""
    sl = slope_through_origin(beta_inc_w, se_inc_w, beta_sev_w, se_sev_w, dilution=dilution)
    z_claim = float(_z(claim_p)); p_t = _p(beta_sev_t / se_sev_t)
    det = dict(slope=sl.as_dict(), p_sev_target=p_t, claim_p=claim_p, alpha=alpha)
    if not np.isfinite(sl.b):
        c = Certificate("UNVERIFIABLE", "S", p_y=p_t, n_units=sl.n_snps, reason="too few control SNPs for a slope")
        return c, c, det
    if p_t > claim_p:
        c = Certificate("NO CLAIM", "S", p_y=p_t, p_aux=sl.p, n_units=sl.n_snps,
                        reason="claim not present at its own threshold in these statistics")
        return c, c, det
    # Slope at which the claim stops clearing its own threshold (the bias standard that actually decides it).
    T = z_claim * se_sev_t
    breakeven_b = float((abs(beta_sev_t) - T) / abs(beta_inc_t)) if beta_inc_t else float("inf")
    # ---- strict (certify_S analogue): NCO arm fires iff slope != 0 at alpha; non-legacy adds the power gate (A2)
    ref = breakeven_b if b_ref is None else float(b_ref)
    mde = float((stats.norm.isf(alpha / 2) + stats.norm.isf(0.2)) * sl.se)
    powered = bool(mde <= ref)
    det.update(strict_b_ref=ref, strict_b_ref_source="breakeven" if b_ref is None else "caller",
               strict_mde_80=mde, strict_powered=powered, strict_rule="legacy" if legacy else "power-gated")
    if sl.p <= alpha:
        sv, swhy = "REJECTED", "strict: slope of severity on selection effects != 0 at alpha"
    elif legacy or powered:
        sv, swhy = "SURVIVES", "strict: claim present AND slope = 0 at alpha" + ("" if legacy else ", arm powered for b_ref")
    else:
        sv, swhy = "UNVERIFIABLE", "strict: slope = 0 at alpha but the arm is under-powered for b_ref (80% MDE > b_ref)"
    strict = Certificate(sv, "S", p_y=p_t, p_aux=sl.p, n_units=sl.n_snps, reason=swhy)
    # ---- targeted: bias on the target B = b * a*, with propagated uncertainty
    B = sl.b * beta_inc_t
    varB = (beta_inc_t ** 2) * sl.se ** 2 + (sl.b ** 2) * se_inc_t ** 2 + sl.se ** 2 * se_inc_t ** 2
    corr = beta_sev_t - B
    se_full = np.sqrt(se_sev_t ** 2 + varB)
    z_point, z_full = corr / se_sev_t, corr / se_full
    same_sign = np.sign(corr) == np.sign(beta_sev_t)
    frac = B / beta_sev_t
    var_frac = varB / beta_sev_t ** 2 + (B ** 2) * se_sev_t ** 2 / beta_sev_t ** 4
    seB = float(np.sqrt(varB))
    # The most claim-favourable corrected effect inside the bias interval (T and breakeven_b are computed above).
    best = corr + Z95 * seB * np.sign(beta_sev_t)
    can_still_pass = bool(np.sign(best) == np.sign(beta_sev_t) and abs(best) >= T)
    det.update(bias=float(B), bias_se=seB, bias_ci=[float(B - Z95 * seB), float(B + Z95 * seB)],
               corrected=float(corr), corrected_se_point=float(se_sev_t), corrected_se_full=float(se_full),
               z_point=float(z_point), z_full=float(z_full), z_claim=z_claim,
               claim_threshold=float(T), corrected_best=float(best), can_still_pass=can_still_pass,
               breakeven_b=breakeven_b, rejected_rule="legacy" if legacy else "symmetric",
               bias_fraction=float(frac), bias_fraction_ci=[float(frac - Z95 * np.sqrt(var_frac)), float(frac + Z95 * np.sqrt(var_frac))])
    rejected = ((not same_sign) or abs(z_point) < z_claim) if legacy else (not can_still_pass)
    if same_sign and abs(z_full) >= z_claim:
        v, why = "SURVIVES", "corrected effect meets the claim threshold with the bias uncertainty propagated"
    elif rejected:
        v, why = "REJECTED", ("point-corrected effect no longer meets the claim threshold" if legacy else
                              "even the most favourable bias in its 95% interval leaves the claim below the threshold")
    else:
        v, why = "UNVERIFIABLE", ("point-corrected effect meets the threshold but the bias interval is too wide to decide"
                                  if legacy else "the bias interval spans the claim threshold")
    targeted = Certificate(v, "S", p_y=p_t, p_aux=_p(z_full), n_units=sl.n_snps, reason="targeted: " + why)
    return targeted, strict, det
