"""proofcarry CLI:
  proofcarry audit   <h5ad>                       -> verification-completeness [S,B,U,D] + S-gate
  proofcarry certify <h5ad> --features F1,F2 --label-col disease --class S [--nco-col smoking]
  proofcarry selftest                             -> synthetic sanity check (no data needed)
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np


def _load(h5ad):
    import anndata as ad
    return ad.read_h5ad(h5ad)


def cmd_audit(args):
    from .manifest import audit
    a = _load(args.h5ad)
    print(json.dumps(audit(a), indent=2))


def cmd_certify(args):
    from .certificate import certify
    a = _load(args.h5ad)
    feats = [f.strip() for f in args.features.split(",") if f.strip()]
    var_names = list(a.var_names)
    idx = [var_names.index(f) for f in feats if f in var_names]
    if not idx:
        print("ERROR: none of the requested features found in var_names", file=sys.stderr); sys.exit(2)
    X = a[:, idx].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    y = (a.obs[args.label_col].astype(str).values != a.obs[args.label_col].astype(str).mode()[0]).astype(int) \
        if args.pos_label is None else (a.obs[args.label_col].astype(str).values == args.pos_label).astype(int)
    unit = a.obs[args.unit_col].astype(str).values
    w = a.obs[args.nco_col].astype(float).values if args.nco_col else None
    batch = a.obs[args.batch_col].astype(str).values if args.batch_col else None
    cert = certify(X, y, unit, confound_class=args.cls, w=w, co_selected=not args.nco_independent,
                   batch=batch, seed=args.seed, rho_ref=args.rho_ref, legacy=args.legacy)
    print(json.dumps(cert.__dict__, indent=2))


def cmd_selftest(args):
    from .certificate import certify
    rng = np.random.default_rng(0)
    # a collider NULL: y independent of composite in population; select on the joint upper tail of
    # C = y + b*comp + b*w so selection induces a reproducible (false) comp~y association.
    n, b, keep = 400, 1.6, 0.3
    N = int(n * 10 / keep)
    comp = rng.standard_normal(N); y = rng.integers(0, 2, N).astype(int); w = rng.standard_normal(N)
    C = y + b * comp + b * w + 0.5 * rng.standard_normal(N)
    sel = np.where(C >= np.quantile(C, 1 - keep))[0][:n]
    comp, y, w = comp[sel], y[sel], w[sel]
    unit = np.arange(n)                                  # one cell per unit for the test
    X = comp[:, None]
    base = certify(X, y, unit, "U", seed=1)              # base severe test: false-certifies the collider
    s_co = certify(X, y, unit, "S", w=w, co_selected=True, seed=1)     # NCO catches it
    s_indep = certify(X, y, unit, "S", w=rng.standard_normal(n), co_selected=False, seed=1)  # refused
    s_none = certify(X, y, unit, "S", w=None, seed=1)    # refused: no NCO retained
    print("SELFTEST (collider NULL):")
    print("  base severe test (U) :", base.verdict, f"(p_y={base.p_y}) -> should FALSE-CERTIFY (SURVIVES)")
    print("  S + co-selected NCO  :", s_co.verdict, f"(p_y={s_co.p_y}, p_yneg={s_co.p_aux}) -> should REJECT")
    print("  S + independent ctrl :", s_indep.verdict, "->", s_indep.reason)
    print("  S + no NCO (real data):", s_none.verdict, "->", s_none.reason)
    ok = (base.verdict == "SURVIVES" and s_co.verdict == "REJECTED"
          and s_indep.verdict == "UNVERIFIABLE" and s_none.verdict == "UNVERIFIABLE")
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="proofcarry")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("audit"); pa.add_argument("h5ad"); pa.set_defaults(func=cmd_audit)
    pc = sub.add_parser("certify")
    pc.add_argument("h5ad"); pc.add_argument("--features", required=True)
    pc.add_argument("--label-col", required=True); pc.add_argument("--pos-label", default=None)
    pc.add_argument("--unit-col", default="donor_id"); pc.add_argument("--class", dest="cls", default="U")
    pc.add_argument("--nco-col", default=None); pc.add_argument("--nco-independent", action="store_true")
    pc.add_argument("--batch-col", default=None); pc.add_argument("--seed", type=int, default=0)
    pc.add_argument("--rho-ref", type=float, default=None,
                    help="class S: the collider signature the NCO arm must be powered to detect (needed for SURVIVES)")
    pc.add_argument("--legacy", action="store_true", help="class S: ungated rule (SURVIVES without a minimum-detectable-effect power check)")
    pc.set_defaults(func=cmd_certify)
    ps = sub.add_parser("selftest"); ps.set_defaults(func=cmd_selftest)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
