# DFT thermodynamic-screen handoff runbook

This runbook takes the current refractory-carbide candidate campaign from
local prerequisites through a provenance-gated Quantum ESPRESSO
energy-above-hull screen. The commands are deliberately split into preparation,
execution, and collection gates so the next operator can stop safely after any
stage.

The workflow does **not** install software or download SSSP, and it never starts
an SCF/relaxation calculation unless the operator supplies `--execute`. The
optional environment probe does launch fixed help/version commands only. A
successful hull result is still a
finite-smearing, non-spin-polarized, scalar-relativistic PBE screening result;
it is not proof of experimental stability or high-temperature performance.

## 0. Operator-owned prerequisites and path variables

Run from the repository root. Replace every `/absolute/path/...` value before
continuing. Use one native installation of QE and MPI for the complete sweep,
confirmation, candidate, and reference campaign; the collectors bind and
compare the QE version, `pw.x` binary hash, MPI launcher hash/version, MPI-rank
count, and OpenMP-thread count.

```bash
PW=/absolute/path/to/native/pw.x
MPI=/absolute/path/to/native/mpirun
PYTHON=${PYTHON:-.conda/bin/python}
SCRATCH=/absolute/path/to/qe_scratch
SSSP_METADATA=/absolute/path/to/SSSP_precision_metadata.json
SSSP_DIR=/absolute/path/to/SSSP_precision_PBE
SSSP_MANIFEST=research/phase_2/dft_validation/sssp_pbe_precision_locked.json

CAMPAIGN=research/phase_2/generation/output/w_c_dft_campaign_v1
SOURCE=$CAMPAIGN/convergence_source_v1
SWEEP=$CAMPAIGN/qe_convergence_sweep_v1
CONFIRM=$CAMPAIGN/qe_convergence_confirmation_v1
CERTIFICATE=$CONFIRM/collected/qe_convergence_certificate.json
CANDIDATE_RELAX=$CAMPAIGN/candidate_relax_certified_v1
CANDIDATE_STATIC=$CANDIDATE_RELAX/static
MP_SNAPSHOT=$CAMPAIGN/mp_snapshot_manifest_v1.json
REFERENCES=$CAMPAIGN/dft_references_v1
HULL=$CAMPAIGN/dft_hull_v1
```

Every preparation and collection output directory below must be new or empty.
The tools fail closed on a non-empty directory so a partial or older campaign
cannot be mixed into the current one. Once a preflight is written, keep the
repository and campaign at the same absolute paths: the provenance records
intentionally bind absolute artifact paths. If the work will run on HPC,
create and prepare the complete campaign on its stable shared filesystem from
the beginning.

Before this runbook can pass, the operator must supply:

- direct, native arm64 or x86_64 executable files for `pw.x` and `mpirun`.
  The current preflight intentionally rejects shell/module wrapper scripts and
  other architectures because their underlying binary identity cannot be
  verified by this handoff code;
- one complete scalar-relativistic SSSP PBE Precision release plus its exact
  official metadata file and reviewed original licenses;
- the existing local `data/mp.2019.04.01.json`,
  `research/phase_2/data/processed/materials.csv`, and
  `research/phase_3/cache/structures.sqlite` files;
- sufficient project and scratch free space. The preflight default is 20 GiB
  on each volume; every executing runner command below also rechecks that
  threshold before each job. A large reference queue is better executed on
  HPC. `$SCRATCH` must already exist as a writable/searchable directory before
  the preflight; the read-only checker deliberately does not create it.

Do not use the broken `/opt/anaconda3/bin/mpif90` toolchain previously found on
this machine. Installation/build instructions are intentionally outside this
code-only handoff.

## 1. Lock and verify the SSSP release

This command reads local files and creates the locked manifest; it performs no
network access or DFT. The acknowledgement flag means the operator actually
reviewed the release's original pseudopotential licenses.

```bash
"$PYTHON" research/phase_2/dft_validation/prepare_sssp_manifest.py \
  --metadata "$SSSP_METADATA" \
  --pseudo-dir "$SSSP_DIR" \
  --library-name "SSSP PBE Precision" \
  --library-version YOUR_EXACT_SSSP_RELEASE \
  --elements C,Ti,V,Zr,Nb,Ta,W \
  --acknowledge-original-licenses \
  --output "$SSSP_MANIFEST"
```

Stop if an official MD5, UPF header, functional, relativistic mode, or cutoff
check fails. `SSSP_MANIFEST` must be a new path: the command refuses to
overwrite an existing locked manifest. Never repair or regenerate that
manifest in the middle of a campaign.

## 2. Run the read-only handoff preflight

The preflight does not scan/hash the multi-gigabyte MP JSON and does not run a
calculation. It checks lightweight dataset headers, disk space, Python plus
importable NumPy/ASE/pymatgen versions, official-metadata MD5 and UPF-header
identity, SSSP hashes, executable hashes, and native architectures. The
explicit probe flag runs only fixed `pw.x -help`, `mpirun --version`, and
`mpirun -np 1 pw.x -help` commands with a five-second timeout. The last probe
verifies launcher/executable interoperability without reading a QE input or
starting an SCF calculation.

```bash
"$PYTHON" research/phase_2/dft_validation/check_qe_handoff_environment.py \
  --project-root "$PWD" \
  --scratch-dir "$SCRATCH" \
  --min-free-gib 20 \
  --pw-executable "$PW" \
  --mpi-executable "$MPI" \
  --mpi-ranks 2 \
  --omp-threads 1 \
  --probe-executables \
  --probe-timeout-seconds 5 \
  --pseudo-manifest "$SSSP_MANIFEST" \
  --pseudo-dir "$SSSP_DIR" \
  --sssp-metadata "$SSSP_METADATA" \
  --raw-json data/mp.2019.04.01.json \
  --materials-csv research/phase_2/data/processed/materials.csv \
  --structure-cache research/phase_3/cache/structures.sqlite \
  --output "$CAMPAIGN/qe_handoff_environment.json"
```

Required gate: exit code `0`, `status=ready`, and an empty `blockers` list.
Exit code `2` is a prerequisite failure, not a Python crash.
For an intentionally serial campaign, pass `--mpi-ranks 1`; in that mode only
`pw.x` is required and the preflight skips the MPI launcher/interoperability
probe. Do not mix serial and MPI execution identities inside one convergence
and production lineage.

## 3. Build a convergence-only source queue

This source queue supplies verified candidate CIF/config/pseudopotential
snapshots to the numerical-convergence study. It is intentionally forbidden
from production execution because it has no convergence certificate yet.

```bash
"$PYTHON" research/phase_2/dft_validation/prepare_qe_jobs.py \
  --validation-report research/phase_2/reports/w_c_structural_validation_v1.json \
  --config research/phase_2/dft_validation/qe_dft_config_v1.json \
  --pseudo-manifest "$SSSP_MANIFEST" \
  --pseudo-dir "$SSSP_DIR" \
  --pw-executable "$PW" \
  --convergence-source-only \
  --top-n 5 \
  --output-dir "$SOURCE"
```

Required gate: `status=convergence_source_only_not_runnable`. That status is
intentional. Do not pass this source preflight directly to an executing
runner. The five queue CIFs are a Git-tracked bundle under
`research/phase_2/dft_validation/candidate_cifs/w_c_structural_validation_v1`;
the report and preparer verify its manifest and per-CIF SHA-256 values.

## 4. Prepare, execute, and collect the sparse convergence sweep

The three current representative IDs cover the full queue element union
`C,Ti,V,Zr,Nb,Ta,W`. If the validation report changes, select representatives
again; the preparer fails closed when their union is incomplete.

```bash
"$PYTHON" research/phase_2/dft_validation/prepare_qe_convergence_jobs.py \
  --source-preflight "$SOURCE/dft_preflight.json" \
  --candidate-id camp_41a80c8b8b30 \
  --candidate-id camp_59538db42ea7 \
  --candidate-id camp_5e6d485d7147 \
  --protocol research/phase_2/dft_validation/qe_convergence_protocol_v1.json \
  --pw-executable "$PW" \
  --scratch-root "$SCRATCH" \
  --output-dir "$SWEEP"

# Dry plan only: no executable is launched.
"$PYTHON" research/phase_2/dft_validation/run_qe_jobs.py \
  --preflight "$SWEEP/convergence_preflight.json" \
  --max-jobs 1 --mpi-ranks 2 --omp-threads 1 \
  --mpi-executable "$MPI"

# Explicitly launches at most one unfinished point. Repeat until complete.
"$PYTHON" research/phase_2/dft_validation/run_qe_jobs.py \
  --preflight "$SWEEP/convergence_preflight.json" \
  --execute --max-jobs 1 --mpi-ranks 2 --omp-threads 1 --min-free-gib 20 \
  --mpi-executable "$MPI" --timeout-seconds 21600

"$PYTHON" research/phase_2/dft_validation/collect_qe_convergence.py \
  --preflight "$SWEEP/convergence_preflight.json" \
  --output-dir "$SWEEP/collected"
```

Required gate: `status=provisional_selection_ready`, complete representative
coverage, and no failed point. `needs_higher_cutoff` or
`needs_denser_kpoint_grid` means the locked tested window must be extended and
rerun. A provisional selection is not authorized for production.

## 5. Confirm the selected combination and issue the certificate

The confirmation stage creates one independent selected-combination SCF per
representative. The certificate is emitted only if all confirmation points
pass component-wise energy, force, and stress thresholds and use the exact
same QE version/binary and execution identity (serial/MPI mode, launcher
hash/version, ranks, and OpenMP threads) as the sweep.

```bash
"$PYTHON" research/phase_2/dft_validation/prepare_qe_convergence_confirmation.py \
  --provisional-summary "$SWEEP/collected/convergence_summary.json" \
  --sweep-preflight "$SWEEP/convergence_preflight.json" \
  --pw-executable "$PW" \
  --scratch-root "$SCRATCH" \
  --output-dir "$CONFIRM"

# Dry plan.
"$PYTHON" research/phase_2/dft_validation/run_qe_jobs.py \
  --preflight "$CONFIRM/confirmation_preflight.json" \
  --max-jobs 1 --mpi-ranks 2 --omp-threads 1 \
  --mpi-executable "$MPI"

# Explicitly launches at most one unfinished confirmation. Repeat as needed.
"$PYTHON" research/phase_2/dft_validation/run_qe_jobs.py \
  --preflight "$CONFIRM/confirmation_preflight.json" \
  --execute --max-jobs 1 --mpi-ranks 2 --omp-threads 1 --min-free-gib 20 \
  --mpi-executable "$MPI" --timeout-seconds 21600

"$PYTHON" research/phase_2/dft_validation/collect_qe_convergence_confirmation.py \
  --preflight "$CONFIRM/confirmation_preflight.json" \
  --output-dir "$CONFIRM/collected"
```

Required gate: `confirmation_summary.json` reports
`status=confirmation_passed`, `$CERTIFICATE` exists, and that certificate
reports `status=passed_tested_window`. Do not edit, relocate after preflight
creation, or recreate the certificate manually; production preparation
re-verifies its canonical payload, absolute artifact paths, and every locked
artifact hash.

The certificate keeps two settings records: the empirically selected setting
and a separate conservative production setting at the strictest joint anchor
actually present in the locked sweep (largest tested cutoff multiplier and
densest tested k-point spacing). Candidate and reference jobs use the latter.
This transfer policy reduces the risk of using a coarse selected point, but it
does **not** prove convergence separately for every unseen structure. Recheck
per-structure convergence for any finalist used in a publication claim.

Trust model: this certificate is a content-integrity receipt, not a digital
signature, identity proof, or external trust anchor. Its hashes detect missing,
mixed, or modified campaign files after collection, but a malicious actor with
write access could re-author both artifacts and hashes. The handoff therefore
assumes a trusted operator, trusted repository commit, and access-controlled
workspace. Preserve the commit ID and, where organizational policy requires
authenticity, sign the commit or immutable handoff archive outside this
workflow.

## 6. Rebuild, run, and collect certified candidate relaxations

Use a fresh directory. The final preparer selects conservative production
settings: cutoffs are at least both the SSSP recommendations and the strictest
tested certificate anchor, while the k-point spacing is no coarser than both
the config and that anchor.

```bash
"$PYTHON" research/phase_2/dft_validation/prepare_qe_jobs.py \
  --validation-report research/phase_2/reports/w_c_structural_validation_v1.json \
  --config research/phase_2/dft_validation/qe_dft_config_v1.json \
  --pseudo-manifest "$SSSP_MANIFEST" \
  --pseudo-dir "$SSSP_DIR" \
  --convergence-certificate "$CERTIFICATE" \
  --pw-executable "$PW" \
  --scratch-root "$SCRATCH" \
  --top-n 5 \
  --output-dir "$CANDIDATE_RELAX"

"$PYTHON" research/phase_2/dft_validation/run_qe_jobs.py \
  --preflight "$CANDIDATE_RELAX/dft_preflight.json" \
  --execute --max-jobs 1 --mpi-ranks 2 --omp-threads 1 --min-free-gib 20 \
  --mpi-executable "$MPI" --timeout-seconds 21600

"$PYTHON" research/phase_2/dft_validation/collect_qe_relaxations.py \
  --preflight "$CANDIDATE_RELAX/dft_preflight.json" \
  --output-dir "$CANDIDATE_RELAX/collected"
```

Repeat the runner command until every row has a bound completed run record.
Required preparation gate: `status=runnable_not_started` and
`production_settings_certified=true`. Required collection gate: every retained
candidate passes electronic/ionic/cell convergence, finite energy, formula,
force, pressure, stress, QE version, executable hash, certified MPI/resource
identity, and certificate lineage.

## 7. Prepare, run, and collect certified candidate static energies

Static SCFs use DFT-relaxed structures and a tighter grid/threshold. They keep
the same convergence-certificate lineage.

```bash
"$PYTHON" research/phase_2/dft_validation/prepare_qe_static_jobs.py \
  --relaxation-results "$CANDIDATE_RELAX/collected/qe_relaxation_results.csv" \
  --relaxation-summary "$CANDIDATE_RELAX/collected/qe_relaxation_summary.json" \
  --relax-preflight "$CANDIDATE_RELAX/dft_preflight.json" \
  --pw-executable "$PW" \
  --scratch-root "$SCRATCH" \
  --output-dir "$CANDIDATE_STATIC"

"$PYTHON" research/phase_2/dft_validation/run_qe_jobs.py \
  --preflight "$CANDIDATE_STATIC/static_preflight.json" \
  --execute --max-jobs 1 --mpi-ranks 2 --omp-threads 1 --min-free-gib 20 \
  --mpi-executable "$MPI" --timeout-seconds 21600

"$PYTHON" research/phase_2/dft_validation/collect_qe_static.py \
  --preflight "$CANDIDATE_STATIC/static_preflight.json" \
  --output-dir "$CANDIDATE_STATIC/collected"
```

Required gate: complete candidate static results with one effective settings
hash, the expected convergence-certificate payload hash, and the same
certified MPI/resource identity.

## 8. Audit the full MP snapshot and materialize competing phases

Unlike the lightweight environment preflight, the snapshot command
intentionally reads/hashes the complete raw JSON, metadata CSV, and SQLite
cache and parses every cached CIF. On the real multi-gigabyte snapshot it can
take a long time. It is required once per immutable snapshot.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
"$PYTHON" research/phase_2/dft_validation/build_mp_snapshot_manifest.py \
  --raw-json data/mp.2019.04.01.json \
  --materials-csv research/phase_2/data/processed/materials.csv \
  --structure-cache research/phase_3/cache/structures.sqlite \
  --output "$MP_SNAPSHOT"

"$PYTHON" research/phase_2/dft_validation/build_mp_reference_inventory.py \
  --candidate-entries "$CANDIDATE_STATIC/collected/candidate_static_energies.csv" \
  --snapshot-manifest "$MP_SNAPSHOT" \
  --materials-csv research/phase_2/data/processed/materials.csv \
  --structure-cache research/phase_3/cache/structures.sqlite \
  --materialize-cifs \
  --output-dir "$REFERENCES"
```

Required gate: exact MP material-ID equality, matching compositions, and an
uncapped, materialized reference inventory for every required non-empty
chemical subsystem. `--diagnostic-max-per-subsystem` and omitting
`--materialize-cifs` are diagnostic modes and cannot unlock hull values.

Before preparing the reference QE queue, inspect its row count and reassess
capacity with `df -h "$PWD" "$SCRATCH"`. The runner checks both the job and
scratch filesystems before every launch and records the measurements, but it
cannot predict the total storage required by the full reference campaign.

## 9. Run references through the identical certified relax/static workflow

The reference queue can be much larger than the candidate queue. Keep the
local safety cap at one job. For HPC, prepare the entire fresh campaign there
from the start; do not move an already-prepared absolute-path-bound campaign.

```bash
"$PYTHON" research/phase_2/dft_validation/prepare_qe_reference_relax_jobs.py \
  --reference-inventory "$REFERENCES/reference_inventory.csv" \
  --config research/phase_2/dft_validation/qe_dft_config_v1.json \
  --pseudo-manifest "$SSSP_MANIFEST" \
  --pseudo-dir "$SSSP_DIR" \
  --convergence-certificate "$CERTIFICATE" \
  --pw-executable "$PW" \
  --scratch-root "$SCRATCH" \
  --output-dir "$REFERENCES/relax"

"$PYTHON" research/phase_2/dft_validation/run_qe_jobs.py \
  --preflight "$REFERENCES/relax/dft_preflight.json" \
  --execute --max-jobs 1 --mpi-ranks 2 --omp-threads 1 --min-free-gib 20 \
  --mpi-executable "$MPI" --timeout-seconds 21600

"$PYTHON" research/phase_2/dft_validation/collect_qe_relaxations.py \
  --preflight "$REFERENCES/relax/dft_preflight.json" \
  --output-dir "$REFERENCES/relax/collected"

"$PYTHON" research/phase_2/dft_validation/prepare_qe_static_jobs.py \
  --relaxation-results "$REFERENCES/relax/collected/qe_relaxation_results.csv" \
  --relaxation-summary "$REFERENCES/relax/collected/qe_relaxation_summary.json" \
  --relax-preflight "$REFERENCES/relax/dft_preflight.json" \
  --pw-executable "$PW" \
  --scratch-root "$SCRATCH" \
  --output-dir "$REFERENCES/static"

"$PYTHON" research/phase_2/dft_validation/run_qe_jobs.py \
  --preflight "$REFERENCES/static/static_preflight.json" \
  --execute --max-jobs 1 --mpi-ranks 2 --omp-threads 1 --min-free-gib 20 \
  --mpi-executable "$MPI" --timeout-seconds 21600

"$PYTHON" research/phase_2/dft_validation/collect_qe_static.py \
  --preflight "$REFERENCES/static/static_preflight.json" \
  --output-dir "$REFERENCES/static/collected"
```

Required gate: complete reference coverage, the same certified lineage as the
candidate static set, and no missing or failed reference phase.

Scratch is intentionally not deleted automatically. Remove a stage-specific
scratch namespace only after its collector gate has passed and its preflight,
queue, inputs, outputs, run records, and collected artifacts are archived.
Never remove scratch belonging to a running or not-yet-collected stage.

## 10. Compute the provenance- and coverage-gated hull

```bash
"$PYTHON" research/phase_2/dft_validation/compute_qe_hull.py \
  --candidate-entries "$CANDIDATE_STATIC/collected/candidate_static_energies.csv" \
  --candidate-summary "$CANDIDATE_STATIC/collected/candidate_static_summary.json" \
  --reference-entries "$REFERENCES/static/collected/reference_static_energies.csv" \
  --reference-summary "$REFERENCES/static/collected/reference_static_summary.json" \
  --reference-inventory "$REFERENCES/reference_inventory.csv" \
  --coverage-manifest "$REFERENCES/reference_coverage_v1.json" \
  --near-hull-threshold-ev-per-atom 0.025 \
  --output-dir "$HULL"
```

Exit code `0` means the audit allowed coverage-complete hull values. Exit code
`2` means audit outputs were written but settings, certificate lineage,
snapshot, inventory, or competing-phase coverage blocked the values; it is not
permission to use blank/diagnostic values. Preserve the user-supplied 0.025
eV/atom threshold in any report rather than relabeling it as a universal
thermodynamic criterion.

## 11. Code-only regression check and handoff bundle

The test suite uses synthetic temporary artifacts and never launches QE:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
XDG_CACHE_HOME="${TMPDIR:-/tmp}/ai-material-cache" \
MPLCONFIGDIR="${TMPDIR:-/tmp}/ai-material-mpl" \
"$PYTHON" -m unittest discover \
  -s research/phase_2/dft_validation -p 'test_*.py' -v
```

Hand the next operator all of the following together:

- the repository commit/branch and this runbook;
- the tracked structural-validation JSON/Markdown report plus the five-CIF
  candidate bundle and SHA-256 manifest under
  `candidate_cifs/w_c_structural_validation_v1`;
- the environment report, locked SSSP manifest, official metadata, and exact
  UPF directory (subject to their original licenses);
- sweep and confirmation preflights, queues, QE inputs/outputs, run records,
  collector CSV/JSON files, and the unedited certificate;
- certified candidate and reference relax/static preflights, inputs, outputs,
  run records, and collector reports;
- MP snapshot manifest, materialized reference inventory/coverage manifest,
  and final hull audit outputs.

Do not combine files from separate output directories or reruns. The hashes
are intended to make that fail closed.

## Scope boundary after the hull

This runbook stops at a zero-K-style DFT formation-energy/hull screen under the
declared approximate electronic protocol. Subsequent claims require separately
specified and validated work:

1. spin-polarized reruns for surviving finalists;
2. spin-orbit sensitivity for Ta/W-containing finalists;
3. phonons for dynamical stability and, if needed, vibrational free energies;
4. elastic constants/mechanical stability;
5. finite-temperature phase stability, oxidation/environmental stability, and
   service-condition models;
6. a licensed, traceable experimental thermal-property dataset for any model
   claiming melting point, thermal conductivity, creep, or related labels.

An LLM can help parse requirements or literature metadata, but it cannot serve
as the missing experimental thermal label or replace DFT/experimental
validation.
