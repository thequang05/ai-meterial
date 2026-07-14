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

## Supplying pseudopotentials

1. Obtain one complete SSSP PBE Precision release from Materials Cloud.
2. Review the licenses/attribution for the selected UPF files.
3. Run `prepare_sssp_manifest.py` against the official release metadata and
   extracted UPFs; do not hand-edit the documentation-only template into a
   runnable manifest.
4. Preserve the official metadata file beside the locked manifest. Job
   preparation re-hashes it, checks official MD5/cutoffs, parses every UPF
   header again, and rejects PBEsol/PBE0 or non-scalar files.
5. Re-run job preparation with that generated manifest and `--pseudo-dir`.

Example (still does not execute DFT):

```bash
.conda/bin/python research/phase_2/dft_validation/prepare_qe_jobs.py \
  --validation-report research/phase_2/reports/w_c_structural_validation_v1.json \
  --config research/phase_2/dft_validation/qe_dft_config_v1.json \
  --pseudo-manifest /absolute/path/sssp_pbe_precision_v2_manifest.json \
  --pseudo-dir /absolute/path/sssp_pbe_precision_v2 \
  --pw-executable /absolute/path/pw.x \
  --top-n 5 \
  --output-dir research/phase_2/generation/output/w_c_campaign_v1/dft_qe_top5_v1
```

`status=runnable_not_started` means inputs and local engine passed preflight;
it does not mean any calculation ran.

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
- `prepare_qe_jobs.py`: writes candidate `vc-relax` inputs only after the UPFs
  pass verification.
- `prepare_qe_convergence_jobs.py`: creates a sparse static-SCF cutoff-pair and
  k-grid sweep on explicitly selected representative structures. The checked-in
  v1 protocol uses eight points per representative instead of a full 4 x 5
  Cartesian grid. It never runs QE.
- `collect_qe_convergence.py`: verifies the locked provenance, parses
  component-wise energy/force/stress changes, applies a stable-tail rule, and
  reports the worst-case setting across representatives. Its result is
  deliberately provisional and cannot serve as a production certificate.
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

## Commands to run later, after QE and SSSP exist locally

The paths under `/absolute/path/...` are placeholders. These commands are
documented for a later machine setup; they have not been run here.

### 1. Lock the local SSSP release

```bash
.conda/bin/python research/phase_2/dft_validation/prepare_sssp_manifest.py \
  --metadata /absolute/path/SSSP_precision_metadata.json \
  --pseudo-dir /absolute/path/SSSP_precision_PBE \
  --library-version YOUR_EXACT_RELEASE \
  --elements C,Ti,V,Zr,Nb,Ta,W \
  --acknowledge-original-licenses \
  --output research/phase_2/dft_validation/sssp_pbe_precision_locked.json
```

### 2. Rebuild the candidate relaxation queue with verified inputs

Use a new output directory instead of mixing it with the earlier blocked
plan-only artifact.

```bash
.conda/bin/python research/phase_2/dft_validation/prepare_qe_jobs.py \
  --validation-report research/phase_2/reports/w_c_structural_validation_v1.json \
  --config research/phase_2/dft_validation/qe_dft_config_v1.json \
  --pseudo-manifest research/phase_2/dft_validation/sssp_pbe_precision_locked.json \
  --pseudo-dir /absolute/path/SSSP_precision_PBE \
  --pw-executable /absolute/path/pw.x \
  --top-n 5 \
  --output-dir research/phase_2/generation/output/w_c_campaign_v1/dft_qe_top5_v2
```

Before launching those relaxation inputs, prepare the sparse convergence study.
The three representatives below cover the complete element union
`C,Ti,V,Zr,Nb,Ta,W` in the current five-candidate report. If the report changes,
choose explicit IDs again; preparation fails closed when their union does not
cover every element in the source queue.

```bash
.conda/bin/python research/phase_2/dft_validation/prepare_qe_convergence_jobs.py \
  --source-preflight research/phase_2/generation/output/w_c_campaign_v1/dft_qe_top5_v2/dft_preflight.json \
  --candidate-id camp_41a80c8b8b30 \
  --candidate-id camp_59538db42ea7 \
  --candidate-id camp_5e6d485d7147 \
  --protocol research/phase_2/dft_validation/qe_convergence_protocol_v1.json \
  --pw-executable /absolute/path/pw.x \
  --output-dir research/phase_2/generation/output/w_c_campaign_v1/qe_convergence_v1
```

This creates 24 static SCF points (eight per representative). Inspect the plan
first; no executable is resolved or launched without `--execute`:

```bash
.conda/bin/python research/phase_2/dft_validation/run_qe_jobs.py \
  --preflight research/phase_2/generation/output/w_c_campaign_v1/qe_convergence_v1/convergence_preflight.json \
  --max-jobs 1 --mpi-ranks 2 --omp-threads 1
```

At home, after QE exists, repeat this explicit command until all 24 points have
a bound completed run record:

```bash
.conda/bin/python research/phase_2/dft_validation/run_qe_jobs.py \
  --preflight research/phase_2/generation/output/w_c_campaign_v1/qe_convergence_v1/convergence_preflight.json \
  --execute --max-jobs 1 --mpi-ranks 2 --omp-threads 1 \
  --timeout-seconds 21600
```

Then collect the tested window:

```bash
.conda/bin/python research/phase_2/dft_validation/collect_qe_convergence.py \
  --preflight research/phase_2/generation/output/w_c_campaign_v1/qe_convergence_v1/convergence_preflight.json \
  --output-dir research/phase_2/generation/output/w_c_campaign_v1/qe_convergence_v1/collected
```

Interpret the status strictly:

- `blocked_incomplete_or_failed_runs`: complete or repair failed points;
- `needs_higher_cutoff`: extend the cutoff window;
- `needs_denser_kpoint_grid`: extend the grid window;
- `provisional_selection_ready`: the sparse sweep found a candidate setting,
  but an independent selected-combination confirmation is still mandatory.

The collector always writes `certificate_eligible=false` and
`confirmation_required=true`. Do not manually copy provisional values into
production inputs. Confirmation/certificate integration is the next code gate.

First inspect a dry plan; this does not resolve or launch MPI:

```bash
.conda/bin/python research/phase_2/dft_validation/run_qe_jobs.py \
  --preflight research/phase_2/generation/output/w_c_campaign_v1/dft_qe_top5_v2/dft_preflight.json \
  --max-jobs 1 --mpi-ranks 2 --omp-threads 1
```

Only the following explicit command starts one unfinished job. Re-running it
skips a previously completed job and advances to the next unfinished job:

```bash
.conda/bin/python research/phase_2/dft_validation/run_qe_jobs.py \
  --preflight research/phase_2/generation/output/w_c_campaign_v1/dft_qe_top5_v2/dft_preflight.json \
  --execute --max-jobs 1 --mpi-ranks 2 --omp-threads 1 \
  --timeout-seconds 21600
```

### 3. Gate relaxation and prepare candidate static energies

```bash
.conda/bin/python research/phase_2/dft_validation/collect_qe_relaxations.py \
  --preflight research/phase_2/generation/output/w_c_campaign_v1/dft_qe_top5_v2/dft_preflight.json \
  --output-dir research/phase_2/generation/output/w_c_campaign_v1/dft_qe_top5_v2/relax_collected

.conda/bin/python research/phase_2/dft_validation/prepare_qe_static_jobs.py \
  --relaxation-results research/phase_2/generation/output/w_c_campaign_v1/dft_qe_top5_v2/relax_collected/qe_relaxation_results.csv \
  --relax-preflight research/phase_2/generation/output/w_c_campaign_v1/dft_qe_top5_v2/dft_preflight.json \
  --pw-executable /absolute/path/pw.x \
  --output-dir research/phase_2/generation/output/w_c_campaign_v1/dft_qe_top5_v2/static

.conda/bin/python research/phase_2/dft_validation/run_qe_jobs.py \
  --preflight research/phase_2/generation/output/w_c_campaign_v1/dft_qe_top5_v2/static/static_preflight.json \
  --execute --max-jobs 1 --mpi-ranks 2 --omp-threads 1

.conda/bin/python research/phase_2/dft_validation/collect_qe_static.py \
  --preflight research/phase_2/generation/output/w_c_campaign_v1/dft_qe_top5_v2/static/static_preflight.json \
  --output-dir research/phase_2/generation/output/w_c_campaign_v1/dft_qe_top5_v2/static/collected
```

### 4. Lock the local MP snapshot and build competing-phase references

First prove that the raw MP-2019 JSON, processed metadata, and structure cache
refer to exactly the same dataset. This step reads and hashes the full raw JSON,
the metadata CSV, the SQLite file, and parses every cached CIF. It can therefore
take a long time on the real 3.8 GB snapshot, but it uses streaming/cursor-based
processing rather than loading the raw JSON or all CIFs into memory. It was not
run during this code-only session.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
.conda/bin/python research/phase_2/dft_validation/build_mp_snapshot_manifest.py \
  --raw-json data/mp.2019.04.01.json \
  --materials-csv research/phase_2/data/processed/materials.csv \
  --structure-cache research/phase_3/cache/structures.sqlite \
  --output research/phase_2/generation/output/w_c_campaign_v1/mp_snapshot_manifest_v1.json
```

The uncapped inventory is required for a coverage-complete screen. Start with
metadata only if you want to inspect its size; add `--materialize-cifs` only
when ready to write all selected reference CIF files. A run without
`--materialize-cifs`, or with `--diagnostic-max-per-subsystem`, is diagnostic
and cannot unlock coverage-complete hull values.

```bash
.conda/bin/python research/phase_2/dft_validation/build_mp_reference_inventory.py \
  --candidate-entries research/phase_2/generation/output/w_c_campaign_v1/dft_qe_top5_v2/static/collected/candidate_static_energies.csv \
  --snapshot-manifest research/phase_2/generation/output/w_c_campaign_v1/mp_snapshot_manifest_v1.json \
  --materials-csv research/phase_2/data/processed/materials.csv \
  --structure-cache research/phase_3/cache/structures.sqlite \
  --materialize-cifs \
  --output-dir research/phase_2/generation/output/w_c_campaign_v1/dft_references_v1
```

Then prepare reference relaxations using the exact candidate snapshots and
bundled UPFs:

```bash
.conda/bin/python research/phase_2/dft_validation/prepare_qe_reference_relax_jobs.py \
  --reference-inventory research/phase_2/generation/output/w_c_campaign_v1/dft_references_v1/reference_inventory.csv \
  --config research/phase_2/generation/output/w_c_campaign_v1/dft_qe_top5_v2/workflow_config_snapshot.json \
  --pseudo-manifest research/phase_2/generation/output/w_c_campaign_v1/dft_qe_top5_v2/pseudo_manifest_snapshot.json \
  --pseudo-dir research/phase_2/generation/output/w_c_campaign_v1/dft_qe_top5_v2/pseudos \
  --pw-executable /absolute/path/pw.x \
  --output-dir research/phase_2/generation/output/w_c_campaign_v1/dft_references_v1/relax
```

Run and collect reference relaxation, then prepare, run, and collect reference
static jobs with the same generic tools. Each runner invocation below launches
at most one unfinished job; repeat the runner command until the queue is
complete. Reference queues can be large, so use an HPC system for production
rather than lifting the local resource cap blindly.

```bash
.conda/bin/python research/phase_2/dft_validation/run_qe_jobs.py \
  --preflight research/phase_2/generation/output/w_c_campaign_v1/dft_references_v1/relax/dft_preflight.json \
  --execute --max-jobs 1 --mpi-ranks 2 --omp-threads 1 \
  --timeout-seconds 21600

.conda/bin/python research/phase_2/dft_validation/collect_qe_relaxations.py \
  --preflight research/phase_2/generation/output/w_c_campaign_v1/dft_references_v1/relax/dft_preflight.json \
  --output-dir research/phase_2/generation/output/w_c_campaign_v1/dft_references_v1/relax/collected

.conda/bin/python research/phase_2/dft_validation/prepare_qe_static_jobs.py \
  --relaxation-results research/phase_2/generation/output/w_c_campaign_v1/dft_references_v1/relax/collected/qe_relaxation_results.csv \
  --relax-preflight research/phase_2/generation/output/w_c_campaign_v1/dft_references_v1/relax/dft_preflight.json \
  --pw-executable /absolute/path/pw.x \
  --output-dir research/phase_2/generation/output/w_c_campaign_v1/dft_references_v1/static

.conda/bin/python research/phase_2/dft_validation/run_qe_jobs.py \
  --preflight research/phase_2/generation/output/w_c_campaign_v1/dft_references_v1/static/static_preflight.json \
  --execute --max-jobs 1 --mpi-ranks 2 --omp-threads 1 \
  --timeout-seconds 21600

.conda/bin/python research/phase_2/dft_validation/collect_qe_static.py \
  --preflight research/phase_2/generation/output/w_c_campaign_v1/dft_references_v1/static/static_preflight.json \
  --output-dir research/phase_2/generation/output/w_c_campaign_v1/dft_references_v1/static/collected
```

The final reference collector outputs are `reference_static_energies.csv` and
`reference_static_summary.json`.

### 5. Compute the coverage-gated hull

```bash
.conda/bin/python research/phase_2/dft_validation/compute_qe_hull.py \
  --candidate-entries research/phase_2/generation/output/w_c_campaign_v1/dft_qe_top5_v2/static/collected/candidate_static_energies.csv \
  --candidate-summary research/phase_2/generation/output/w_c_campaign_v1/dft_qe_top5_v2/static/collected/candidate_static_summary.json \
  --reference-entries research/phase_2/generation/output/w_c_campaign_v1/dft_references_v1/static/collected/reference_static_energies.csv \
  --reference-summary research/phase_2/generation/output/w_c_campaign_v1/dft_references_v1/static/collected/reference_static_summary.json \
  --reference-inventory research/phase_2/generation/output/w_c_campaign_v1/dft_references_v1/reference_inventory.csv \
  --coverage-manifest research/phase_2/generation/output/w_c_campaign_v1/dft_references_v1/reference_coverage_v1.json \
  --near-hull-threshold-ev-per-atom 0.025 \
  --output-dir research/phase_2/generation/output/w_c_campaign_v1/dft_hull_v1
```

Exit code `2` means the audit files were written but reference-hull values are
blocked by incomplete or inconsistent provenance/coverage. It is not a
software crash. Even a complete pass is a screening result under the declared
finite-smearing PBE workflow, not proof of experimental or service-condition
stability. `--near-hull-threshold-ev-per-atom` is explicitly a user-supplied
decision threshold; the output field is therefore named
`within_user_supplied_hull_threshold`, not a thermodynamic-validation label.

## Lightweight code-only verification

This test suite uses temporary synthetic files and never executes QE:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
XDG_CACHE_HOME=/private/tmp/ai-material-cache \
MPLCONFIGDIR=/private/tmp/ai-material-mpl \
.conda/bin/python -m unittest discover \
  -s research/phase_2/dft_validation -p 'test_*.py' -v
```
