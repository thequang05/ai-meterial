# DFT validation preparation (Quantum ESPRESSO)

This directory prepares reproducible DFT jobs for candidates that already
passed chemistry, CIF, GNN, and CHGNet screening. It does **not** treat a
prepared input, a CHGNet-relaxed CIF, or a negative GNN formation-energy
prediction as DFT validation.

## Current scope

The first workflow is locked to:

- Quantum ESPRESSO `pw.x`, `vc-relax`;
- PBE;
- one internally consistent scalar-relativistic SSSP PBE Precision release;
- metallic Marzari–Vanderbilt smearing;
- one global plane-wave/density cutoff across the complete candidate queue;
- an automatic k-grid derived from a fixed reciprocal-space spacing;
- at most six elements per candidate in protocol v1, preventing exponential
  subsystem expansion for malformed/high-entropy inputs;
- one calculation at a time on the local 8 GB Mac.

The official `pw.x` input contract is documented at
<https://www.quantum-espresso.org/Doc/INPUT_PW.html>. SSSP releases and their
recommended cutoffs are available at <https://sssp.materialscloud.org/>.
SSSP is a collection: each pseudopotential retains its original license, so
the exact files, licenses, attribution, release, and hashes must be recorded.

## Safe plan-only command

This command performs no DFT calculation and needs no pseudopotentials:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
.conda/bin/python research/phase_2/dft_validation/prepare_qe_jobs.py \
  --validation-report research/phase_2/reports/w_c_structural_validation_v1.json \
  --config research/phase_2/dft_validation/qe_dft_config_v1.json \
  --top-n 5 \
  --output-dir research/phase_2/generation/output/w_c_campaign_v1/dft_qe_top5_v1
```

Expected status on the current machine:

```text
blocked_missing_pseudopotentials
```

The command still creates an auditable five-candidate queue and copies each
input CIF. It does not emit `vc-relax.in` until all UPF metadata and SHA-256
checks pass.

## Supplying pseudopotentials for the convergence source

1. Obtain one complete SSSP PBE Precision release from Materials Cloud.
2. Review the licenses/attribution for the selected UPF files.
3. Run `prepare_sssp_manifest.py` against the official release metadata and
   extracted UPFs; do not hand-edit the documentation-only template into a
   runnable manifest.
4. Preserve the official metadata file beside the locked manifest. Job
   preparation re-hashes it, checks official MD5/cutoffs, parses every UPF
   header again, and rejects PBEsol/PBE0 or non-scalar files.
5. Re-run job preparation with that generated manifest, `--pseudo-dir`, and
   `--convergence-source-only`. This produces the verified bootstrap queue used
   by the convergence study; it deliberately does not produce a runnable
   `vc-relax` input.

Example (still does not execute DFT):

```bash
.conda/bin/python research/phase_2/dft_validation/prepare_qe_jobs.py \
  --validation-report research/phase_2/reports/w_c_structural_validation_v1.json \
  --config research/phase_2/dft_validation/qe_dft_config_v1.json \
  --pseudo-manifest /absolute/path/sssp_pbe_precision_v2_manifest.json \
  --pseudo-dir /absolute/path/sssp_pbe_precision_v2 \
  --pw-executable /absolute/path/pw.x \
  --convergence-source-only \
  --top-n 5 \
  --output-dir research/phase_2/generation/output/w_c_campaign_v1/dft_qe_top5_v1
```

Expected status is `convergence_source_only_not_runnable`. The confirmation
collector must later issue a valid convergence certificate before a fresh
production preparation can reach `status=runnable_not_started`. Neither status
means that a calculation has already run.

## Scientific gates after preparation

Before production ranking:

1. Converge plane-wave cutoff and k-point spacing on representative carbide
   cells; do not assume the first config is publication-converged.
2. Run one candidate at a time initially and verify electronic/ionic/cell
   convergence plus final stress and forces.
3. Use a separate, tighter static-energy step on each DFT-relaxed structure.
4. Calculate all relevant competing phases with the **same** functional,
   pseudopotential release, cutoffs, smearing policy, and convergence settings.
5. Build composition-specific phase diagrams and energy-above-hull values.
6. Test spin polarization for finalists; assess spin–orbit coupling for
   Ta/W-containing finalists before publication-quality ranking.
7. Only after the thermodynamic gate, proceed to phonons, elastic properties,
   oxidation/environmental stability, and thermal-property evaluation.

CHGNet energy values are not reference energies for a DFT convex hull.

## What is implemented now

All files in this directory are code/preflight tools. No installer and no
automatic downloader is included. Merely running unit tests does not invoke
`pw.x`.

- `prepare_sssp_manifest.py`: verifies official metadata MD5, UPF element,
  PBE/scalar-relativistic headers, cutoffs, and records SHA-256.
- `prepare_qe_jobs.py`: writes production candidate/reference `vc-relax` inputs
  only after both the UPFs and an independently confirmed convergence
  certificate pass full hash verification. `--convergence-source-only` creates
  a deliberately non-runnable bootstrap queue for the convergence study.
- `prepare_qe_convergence_jobs.py`: creates a sparse static-SCF cutoff-pair and
  k-grid sweep on explicitly selected representative structures. The checked-in
  v1 protocol uses eight points per representative instead of a full 4 x 5
  Cartesian grid. It never runs QE.
- `collect_qe_convergence.py`: verifies the locked provenance, parses
  component-wise energy/force/stress changes, applies a stable-tail rule, and
  reports the worst-case setting across representatives. Its result is
  deliberately provisional and cannot serve as a production certificate.
- `prepare_qe_convergence_confirmation.py` and
  `collect_qe_convergence_confirmation.py`: test the selected cutoff/k-point
  combination independently for every representative and emit
  `qe_convergence_certificate.json` only on a complete component-wise pass.
- `qe_convergence_certificate.py`: re-verifies the certificate canonical hash,
  every locked sweep/confirmation artifact, config/protocol/manifest, and each
  UPF before production use. Production jobs use the strictest joint anchor
  in the tested window, not merely the first passing selection; this remains a
  conservative transfer rule and is not proof of per-structure convergence for
  unseen candidates or references.
- `check_qe_handoff_environment.py`: read-only software/storage/dataset/SSSP
  preflight; it never downloads software or launches a DFT calculation.
- `run_qe_jobs.py`: dry-run by default; `--execute` is mandatory, execution is
  sequential, local resources are capped, and a timeout terminates the full
  MPI process group.
- `collect_qe_relaxations.py`: gates QE/ionic/electronic completion, finite
  energy, formula/cell, forces, pressure, and the full residual stress tensor.
- `prepare_qe_static_jobs.py` and `collect_qe_static.py`: create and collect a
  tighter SCF calculation on DFT-relaxed structures. The effective settings
  hash also includes the QE version and executable binary hash.
- `build_mp_snapshot_manifest.py`: streams and hashes the raw MP-2019 JSON,
  verifies exact material-ID equality across raw JSON, processed metadata, and
  the SQLite structure cache, and checks every cached CIF composition against
  metadata.
- `build_mp_reference_inventory.py`: selects every local MP-2019 structure
  from every required non-empty chemical subsystem, but only from a verified
  snapshot manifest. An optional diagnostic cap always marks coverage
  incomplete.
- `prepare_qe_reference_relax_jobs.py`: sends the fixed reference inventory
  through the same relaxation/static workflow as candidates.
- `compute_qe_hull.py`: emits a finite-smearing, non-spin-polarized,
  scalar-relativistic PBE reference-hull screen only if snapshot, inventory,
  structure lineage, collector provenance, settings, and coverage gates all
  pass. Otherwise coverage-complete values remain blank.
- `build_reference_coverage_template.py`: diagnostic subsystem enumeration
  only. Its output is intentionally rejected by `compute_qe_hull.py` and
  cannot replace the verified MP snapshot/inventory workflow.

## Authoritative operator commands

Use [`DFT_HANDOFF_RUNBOOK.md`](DFT_HANDOFF_RUNBOOK.md) for the current complete
sequence from environment/SSSP checks through convergence confirmation,
certificate-gated candidate/reference calculations, and energy-above-hull.
Production preparation fails closed without a valid certificate. Historical
pre-certificate command transcripts were removed to avoid copy/pasting an
obsolete execution path; Git history remains the implementation archive.
