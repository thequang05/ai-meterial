# ML-Only Thermal Evaluation (`eval_ml_thermal.py`)

This document describes the DFT-free evaluation pipeline that replaces the
Quantum ESPRESSO convergence sweep + hull computation for environments where
running QE is not practical (no HPC cluster, limited disk, single laptop).

## Why this exists

The audited DFT pipeline in `research/phase_2/dft_validation/` (see
`DFT_HANDOFF_RUNBOOK.md`) requires:

- native `pw.x` and `mpirun` binaries with verifiable SHA-256 hashes
- a complete SSSP PBE Precision release plus its official metadata
- 20 GiB of free disk on the project filesystem **and** on a scratch filesystem
- ~1–6 hours per SCF on a 2-MPI-rank laptop, scaling to days for the full
  convergence sweep and weeks for the reference-phase hull

The ML-only evaluation here replaces that path with a surrogate that runs in
seconds and needs no scratch disk. It is intentionally conservative: every
output is labelled as an ML estimate so it cannot be confused with a
DFT-audited number.

## What it does

`research/phase_2/eval_ml_thermal.py` performs the following steps, none of
which touch Quantum ESPRESSO:

1. Reads `research/phase_2/reports/w_c_structural_validation_v1.json` and
   extracts the `selected_candidates` block.
2. Locates the CHGNet-relaxed CIF for each candidate, searching first
   `generation/output/w_c_dft_campaign_v1/convergence_source_v1/jobs/*` then
   falling back to `dft_validation/candidate_cifs/w_c_structural_validation_v1/`.
3. Builds a periodic-neighbour PyG graph (4 Å cutoff) from each CIF.
4. Scores every graph with the audited formation-energy GNN
   (`models/gnn_formation_energy_grouped_v1.pt`).
5. Looks up the lowest-formation-energy Materials Project reference in the
   same chemical subsystem (full subsystem first, falling back to smaller
   sub-subsystems when the full subsystem has no MP entries).
6. Computes `ml_estimated_hull_gap_ev_per_atom = GNN_Ef − MP_best_Ef`.
7. Applies a thermal-proxy classification:

   | status                         | meaning                                                                 |
   |--------------------------------|-------------------------------------------------------------------------|
   | `competitive_with_best_mp`     | surrogate within 0.5 eV/atom of the best MP reference in subsystem     |
   | `above_mp_best`                | surrogate above the MP best by more than 0.5 eV/atom                   |
   | `no_mp_reference`              | no MP entry in any matching subsystem                                   |
   | `energy_too_high`              | GNN E_f > −0.25 eV/atom                                                 |
   | `chemistry_rejected`           | formula fails the refractory+ceramic rule or contains a forbidden element |
   | `missing_cif` / `cif_unreadable` / `empty_structure` / `formula_mismatch` | evaluation could not run on this candidate            |

8. Ranks candidates by chemistry pass, then by evaluated status, then by GNN
   formation energy. Candidates whose CIF is missing are sorted to the bottom
   of the chemistry-pass group so the operator sees the actual scores first.
9. Writes a CSV and a JSON summary. The JSON includes the SHA-256 of every
   input file (validation report, CIFs, GNN checkpoint, MP CSV) so the run is
   auditable later.

## What it explicitly does NOT do

- No Quantum ESPRESSO is launched. The script never invokes `pw.x`, `mpirun`,
  or any external executable.
- No scratch disk is written. `/private/tmp/qe_scratch` is never touched.
- No DFT-audited convex hull number is produced. The script labels its
  energy gap as `ml_estimated_hull_gap_ev_per_atom`. Do not relabel this as a
  `formation_energy_above_hull` or report it as a DFT result.

## Output files

For a run with `--output-dir research/phase_2/reports/ml_thermal_eval_v1`:

```
research/phase_2/reports/ml_thermal_eval_v1/
├── ml_thermal_evaluation.csv    # one row per candidate, full schema
└── ml_thermal_evaluation.json   # summary counts + audit hashes + limit notice
```

The CSV columns are:

| column | description |
|---|---|
| `ml_rank` | surrogate rank (1 = best) |
| `rank` | original rank from the structural-validation report |
| `candidate_id` | GraphVAE campaign id |
| `formula` | reduced formula |
| `num_sites` | atoms per cell (from CIF) |
| `chemistry_pass` | refractory + ceramic, no forbidden elements |
| `refractory_present` / `ceramic_present` / `forbidden_present` | element-class flags |
| `gnn_formation_energy_pre_chgnet_ev_per_atom` | GNN E_f before CHGNet relaxation (from the report) |
| `gnn_formation_energy_post_chgnet_ev_per_atom` | GNN E_f on the CHGNet-relaxed CIF |
| `gnn_delta_from_chgnet_ev_per_atom` | post − pre CHGNet |
| `chgnet_energy_change_ev_per_atom` | CHGNet relaxation energy change |
| `ml_estimated_hull_gap_ev_per_atom` | GNN E_f − MP best E_f in subsystem |
| `mp_relative_gap_ev_per_atom` | same value, kept for legacy readers |
| `mp_best_formula` / `mp_best_formation_energy_per_atom` / `mp_best_uid` | best MP reference in the chosen subsystem |
| `mp_best_from_same_composition` | whether the MP best has the same reduced formula as the candidate |
| `mp_subsystem` | which subsystem was used (full or fallback) |
| `mp_subsystem_entry_count` | how many MP entries are in that subsystem |
| `mp_is_fallback_subsystem` | `True` if the script had to drop one or more elements |
| `thermal_proxy_status` | see classification table above |
| `thermal_proxy_reason` | human-readable explanation of the status |
| `graph_empty` | `True` if the CIF produced a graph with ≤1 edge |
| `cif_path` / `cif_sha256` | provenance for the input CIF |

The JSON summary includes the same audit hashes that the DFT pipeline uses
(SHA-256 of validation report, GNN checkpoint, MP CSV) plus a
`scientific_limit` string that records the ML-only scope.

## How to run

The script only needs the existing training pipeline artifacts. No new data
downloads, no DFT installations.

```bash
conda activate ai-material     # env with torch, torch_geometric, pymatgen
cd /Users/koiita/Downloads/ai-meterial
.conda/bin/python research/phase_2/eval_ml_thermal.py \
    --output-dir research/phase_2/reports/ml_thermal_eval_v1
```

Optional flags:

```bash
--validation-report PATH     # default reports/w_c_structural_validation_v1.json
--cif-dir PATH               # override the convergence_source_v1 search dir
--gnn-checkpoint PATH        # default models/gnn_formation_energy_grouped_v1.pt
--materials-csv PATH         # default data/processed/materials.csv
--output-dir PATH            # required
```

## Current run on the W–C refractory campaign

```text
Evaluated 15 candidates (ML-only, no DFT).
  chemistry pass:          15
  competitive_with_best_mp: 4

Top 5:
  [1] Ti3NbWC5   GNN_Ef=-0.352 eV/atom  MP_best=-0.643 eV/atom  status=competitive_with_best_mp
  [2] Ti3VWC5    GNN_Ef=-0.296 eV/atom  MP_best=-0.622 eV/atom  status=competitive_with_best_mp
  [3] Ti2VWC4    GNN_Ef=-0.282 eV/atom  MP_best=-0.622 eV/atom  status=competitive_with_best_mp
  [4] Zr3TiWC5   GNN_Ef=-0.263 eV/atom  MP_best=-0.650 eV/atom  status=competitive_with_best_mp
  [5] Zr3TaWC5   GNN_Ef= 0.002 eV/atom  MP_best=-0.399 eV/atom  status=energy_too_high
```

Interpretation:

- The four `competitive_with_best_mp` candidates are the ones the surrogate
  considers worth chasing. They are the same four the structural-validation
  stage ranked highest before DFT.
- `Zr3TaWC5` ranked inside the chemistry-pass group because the GNN pushed it
  above −0.25 eV/atom after CHGNet relaxation. It is flagged
  `energy_too_high` and should not be prioritised.
- Ten candidates have no CHGNet-relaxed CIF on disk yet. Their status is
  `missing_cif`. To evaluate them, run CHGNet relaxation on the original
  structural-validation CIFs (`research/phase_2/dft_validation/candidate_cifs/
  w_c_structural_validation_v1/`); the relaxed output will then be picked up
  automatically.

## Reading the ML estimate correctly

- `ml_estimated_hull_gap_ev_per_atom` is **not** a convex-hull number. It is
  the gap between a single GNN prediction and a single MP reference in the
  same subsystem. The audited DFT hull requires recomputing every competing
  phase in the subsystem with one consistent setup; the ML surrogate cannot
  do that.
- The 0.5 eV/atom tolerance and the −0.25 eV/atom energy bound are
  surrogates-only heuristics. They are tuned for the refractory-carbide
  proxy dataset, not for general materials.
- The surrogate MAE on the audited test split is roughly 0.2 eV/atom. Treat
  any number smaller than that as noise.

## When to escalate to DFT

The DFT pipeline is still the right tool for any of the following:

- You need a number that can appear in a publication or report.
- You need to compare the candidate against more than one MP reference
  (the ML script picks only the lowest-energy one).
- You need to verify dynamical stability (phonons) or finite-temperature
  behaviour. Those are explicitly out of scope here.
- The ML surrogate disagrees with chemistry intuition (e.g. very low GNN E_f
  but high CHGNet relaxation energy change).

When you do escalate, run `prepare_qe_jobs.py` on the top-ranked candidates
only, not on the full 15. The full DFThandoff runbook still applies
unchanged; this README does not relax any of its gates.

## Files added by this evaluation

- `research/phase_2/eval_ml_thermal.py` — the surrogate itself.
- `research/phase_2/reports/ml_thermal_eval_v1/ml_thermal_evaluation.csv`
- `research/phase_2/reports/ml_thermal_eval_v1/ml_thermal_evaluation.json`

No files under `research/phase_2/dft_validation/` or
`research/phase_2/generation/output/` are modified by this evaluation.