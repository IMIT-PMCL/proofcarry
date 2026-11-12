"""Proof-carrying-data manifest: the auxiliary variables a class-matched verifier requires.

A dataset is *verification-complete* for a confound class only if it retains that class's minimal
auxiliary variable (verification_completeness_theorem.md). This module defines the manifest that
travels with the data (in `adata.uns['proof_carrying']`), declaring, per class {S,B,U,D}, which
obs column instantiates the auxiliary — so a downstream verifier can REFUSE an un-certifiable test
instead of silently running it.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict

UNS_KEY = "proof_carrying"

# per-class minimal auxiliary (what makes the class certifiable at all)
CLASS_AUX = {
    "S": "co-selected negative-control outcome (NCO) or a recorded selection/ascertainment variable",
    "B": "processing-run / batch indicator (lane, plate, sequencing run), distinct from the biological group",
    "U": "donor / subject / unit id (for donor-level pseudobulk, not cell-as-replicate)",
    "D": "covariate-shift variables (age, sex, ancestry)",
}


@dataclass
class NCO:
    """A declared co-selected negative-control outcome. `expected_null` = the pre-registered
    expectation X ⫫ this-outcome | Y under the true-effect model (so a within-cohort association
    flags collider selection). `co_selected=True` asserts it passed through the same selection
    channel as the claimed outcome (the RELEVANCE condition — a valid-but-independent control does
    NOT certify; see Supplementary Information §S2)."""
    name: str
    obs_col: str
    co_selected: bool = True
    rationale: str = ""


@dataclass
class ProofCarryingManifest:
    unit_col: Optional[str] = None            # U
    batch_col: Optional[str] = None           # B
    selection_col: Optional[str] = None       # S (recorded selection/ascertainment variable)
    nco: List[NCO] = field(default_factory=list)   # S (co-selected negative-control outcomes)
    demographics: List[str] = field(default_factory=list)   # D
    notes: str = ""

    # ---- verification-completeness scoring (0 foreclosed / 1 conditional / 2 guaranteed) ----
    def completeness(self) -> Dict[str, int]:
        s = 2 if any(n.co_selected for n in self.nco) else (1 if (self.nco or self.selection_col) else 0)
        return {
            "S": s,
            "B": 2 if self.batch_col else 0,
            "U": 2 if self.unit_col else 0,
            "D": 2 if len(self.demographics) >= 2 else (1 if self.demographics else 0),
        }

    def s_certifiable(self) -> bool:
        """The proof-carrying-data gate: is a selection-collider certificate even DEFINED here?"""
        return any(n.co_selected for n in self.nco)

    def to_uns(self) -> dict:
        d = asdict(self)
        # h5ad/anndata's uns writer cannot serialize a nested list-of-dicts (raises TypeError:
        # "Can't implicitly convert non-string objects to strings" when it tries to write `nco` as a
        # vlen string array). JSON-encode it to a single string so the manifest actually round-trips
        # through a real .h5ad file; from_uns() below decodes it back transparently. Found + fixed
        # 2026-09-25 while building the first real adoption demo (docs/theory/proofcarry_adoption_demo.md)
        # -- this was previously untested against an actual h5ad write.
        d["nco"] = json.dumps(d["nco"])
        d["verification_completeness"] = self.completeness()
        d["s_certifiable"] = self.s_certifiable()
        d["_schema"] = "proofcarry/0.1"
        return d

    @staticmethod
    def from_uns(d: dict) -> "ProofCarryingManifest":
        nco_raw = d.get("nco", [])
        if isinstance(nco_raw, str):
            nco_raw = json.loads(nco_raw) if nco_raw else []
        ncos = [NCO(**n) if isinstance(n, dict) else n for n in nco_raw]
        return ProofCarryingManifest(
            unit_col=d.get("unit_col"), batch_col=d.get("batch_col"),
            selection_col=d.get("selection_col"), nco=ncos,
            demographics=list(d.get("demographics", [])), notes=d.get("notes", ""))


def write_manifest(adata, m: ProofCarryingManifest) -> None:
    adata.uns[UNS_KEY] = m.to_uns()


def read_manifest(adata) -> Optional[ProofCarryingManifest]:
    d = adata.uns.get(UNS_KEY) if hasattr(adata, "uns") else None
    return ProofCarryingManifest.from_uns(d) if d else None


def audit(adata) -> dict:
    """Verification-completeness audit of an AnnData: infer what auxiliaries the obs schema carries,
    return the [S,B,U,D] vector + whether an S-certificate is defined. Uses the manifest if present,
    else best-effort inference from obs column names."""
    m = read_manifest(adata)
    inferred = False
    if m is None:
        inferred = True
        cols = list(getattr(adata, "obs").columns) if hasattr(adata, "obs") else []
        low = {c.lower(): c for c in cols}
        def find(keys):
            for k in keys:
                for lc, orig in low.items():
                    if k in lc:
                        return orig
            return None
        unit = find(["donor", "subject", "individual", "patient", "sample_id"])
        batch = find(["batch", "library", "lane", "plate", "run", "10x", "pool"])
        sel = find(["selection", "recruit", "admission", "severity", "who_", "qc_pass", "retained"])
        demo = [find([k]) for k in ("age", "sex", "ethnic", "ancestr")]
        m = ProofCarryingManifest(unit_col=unit, batch_col=batch, selection_col=sel,
                                  demographics=[d for d in demo if d])
    return {"verification_completeness": m.completeness(),
            "s_certifiable": m.s_certifiable(),
            "manifest_source": "inferred_from_obs" if inferred else "declared",
            "unit_col": m.unit_col, "batch_col": m.batch_col,
            "selection_col": m.selection_col,
            "nco": [n.name for n in m.nco],
            "message": ("S-certificate DEFINED (a co-selected NCO is retained)." if m.s_certifiable()
                        else "S-FORECLOSED: no co-selected NCO retained — a selection-collider "
                             "certificate is UNDEFINED for this dataset (proof-carrying-data gap).")}
