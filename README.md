# proofcarry

**Bias-matched statistical certificates and proof-carrying-data manifests for scientific discovery.**

Reference implementation accompanying the manuscript:
> *Information loss in curated data bounds human and AI scientific discovery*

---

## Overview

Standard internal validation (e.g., cell-level permutation and false-discovery-rate control) is structurally blind to patient selection bias and technical confounding when the required auxiliary metadata have been stripped during curation. Under the **verifiability dissipation law**, a scientific claim remains testable against a confounding family if and only if the curated dataset retains that family's matched minimal control variable.

`proofcarry` provides:
1. **Bias-matched statistical certificates (`proofcarry.certify`, `proofcarry.sumstat`)** across individual-level matrices (`AnnData` / NumPy) and genome-wide association summary statistics (`certify_S_sumstat`), covering:
   - **`S` (Selection / collider bias)**: Co-selected negative-control outcome (NCO) equivalence and power-aware minimal detectable selection bias (`rho_min`, `breakeven_b`, `strict_mde_80`).
   - **`B` (Technical batch confounding)**: Within-batch stratified permutation certificate.
   - **`U` (Repeated cell/sample measurements)**: Biological donor/unit-level aggregation permutation certificate.
2. **Refusal of unverifiable claims**: Rather than silently certifying an underpowered or control-stripped dataset, `proofcarry` explicitly emits `UNVERIFIABLE` when the matched control variable is missing, not co-selected, or statistically underpowered for the declared reference signature (`rho_ref` / `b_ref`).
3. **Proof-Carrying Data Manifest (`ProofCarryingManifest`, `proofcarry audit`)**: A lightweight JSON-schema metadata contract (`adata.uns['proof_carrying']`) that records the retained verification controls and scores any dataset's verification completeness across `[S, B, U, D]`.

---

## System Requirements & Installation

- **Python**: `>= 3.9`
- **Core dependencies**: `numpy >= 1.21`, `scipy >= 1.7`
- **Optional dependencies**: `anndata >= 0.8` (for `.h5ad` CLI audit and certification)

Install in editable or standard mode (typical installation time `< 10 seconds`):

```bash
pip install -e .
# Optional: include AnnData support for .h5ad files
pip install -e ".[anndata]"
```

---

## Quick Verification (No External Data Required)

Run the built-in synthetic self-test and unit test suite (`< 5 seconds` total runtime):

```bash
proofcarry selftest
python3 tests/test_power_branch.py
python3 tests/test_strict_power.py
python3 tests/test_targeted_symmetry.py
```

Expected output from `proofcarry selftest`:
```text
SELFTEST (collider NULL):
  base severe test (U) : SURVIVES (p_y=0.0005) -> should FALSE-CERTIFY (SURVIVES)
  S + co-selected NCO  : REJECTED (p_y=0.0005, p_yneg=0.0) -> should REJECT
  S + independent ctrl : UNVERIFIABLE -> NCO is not co-selected ...
  S + no NCO (real data): UNVERIFIABLE -> no negative-control outcome supplied
RESULT: PASS
```

---

## Python API Usage

### 1. Individual-Level / Single-Cell Certification (`proofcarry.certify`)

```python
import proofcarry as pc

# Certify a candidate claim against selection bias (class "S") using a co-selected negative control w
cert = pc.certify(
    X, y, unit=donor_id,
    confound_class="S",
    w=co_selected_nco,
    co_selected=True,
    rho_ref=0.25,
)
print(cert)
# -> Certificate(SURVIVES | REJECTED | UNVERIFIABLE)

# Without a co-selected negative control, verification against selection bias is refused:
cert_missing = pc.certify(X, y, unit=donor_id, confound_class="S", w=None)
# -> Certificate(UNVERIFIABLE [S] ...: no negative-control outcome supplied)
```

- **Conditioning guardrail**: Conditioning on categorical outcome $Y$ uses exact within-$Y$-stratum centering rather than linear regression partialling, which is invalid under nonlinear post-selection truncation (Supplementary Information §S2).
- **Power-aware certification**: For class `S`, returning `SURVIVES` requires declaring `rho_ref` (the reference collider correlation the negative-control arm must be powered to detect at 80% power, $\rho_{\min}(n) = 2.486 / \sqrt{n}$). If the negative-control arm does not fire but $\rho_{\min}(n) > \rho_{\text{ref}}$, `proofcarry` returns `UNVERIFIABLE` rather than a false certification.

### 2. GWAS Summary-Statistics Selection Certificate (`proofcarry.sumstat.certify_S_sumstat`)

```python
from proofcarry.sumstat import certify_S_sumstat

# y, se_y: negative-control SNP effects and SEs on the selected trait (e.g., progression/severity)
# x, se_x: negative-control SNP effects and SEs on disease incidence / selection
# beta_out, se_out: target locus effect and SE on the selected trait
# a_sel, se_sel: target locus effect and SE on incidence / selection
targeted_cert, strict_cert, details = certify_S_sumstat(
    y, se_y, x, se_x,
    beta_out, se_out, a_sel, se_sel,
    claim_p=5e-8,
)
print("Targeted verdict:", targeted_cert.verdict)
print("Strict verdict  :", strict_cert.verdict)
```

### 3. Proof-Carrying Data Manifest (`ProofCarryingManifest` & `audit`)

Attach verification metadata directly to an `AnnData` object (`adata.uns['proof_carrying']`):

```python
from proofcarry import ProofCarryingManifest, NCO, write_manifest, audit

manifest = ProofCarryingManifest(
    unit_col="donor_id",
    batch_col="library_id",
    selection_col="hospital_admission",
    nco=[
        NCO(
            name="preexisting_hypertension",
            obs_col="PreExistingHypertension",
            co_selected=True,
            rationale="Shares hospital-admission selection channel; orthogonal to target pathway",
        )
    ],
    demographics=["age", "sex", "self_reported_ethnicity"],
)
write_manifest(adata, manifest)
print(audit(adata))
```

---

## License & Patent Notice

This software is released under an **Academic Research and Non-Commercial License with Explicit Patent Reservation** (see [LICENSE](./LICENSE) and [PATENTS.md](./PATENTS.md)).

- **Academic & Peer-Review Verification**: Free to view, run, and modify for non-commercial academic research and editorial/peer-review evaluation.
- **No Patent License Granted**: The methods and architectures implemented in `proofcarry` are covered by pending patent applications (CNIPA and international PCT priority filings). No express or implied patent license is granted by this repository; all patent and commercial rights are expressly reserved by Ruijin Hospital, Shanghai Jiao Tong University School of Medicine.
- **Commercial Inquiries**: Please contact Prof. Ning Wen (`wn12479@rjh.com.cn`).
