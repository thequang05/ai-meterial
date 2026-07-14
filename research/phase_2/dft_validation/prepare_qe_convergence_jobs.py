"""Prepare sparse, provenance-locked Quantum ESPRESSO convergence sweeps.

This module never launches Quantum ESPRESSO.  It takes explicit representative
structures from an existing candidate relaxation preflight and prepares two
one-dimensional static-SCF sweeps per representative:

* cutoff-pair multipliers at the densest protocol k-grid;
* k-grid spacings at the highest protocol cutoff pair.

The shared high-cutoff/dense-grid anchor is emitted once, so a protocol with
four cutoff levels and five k-grid levels needs eight jobs instead of twenty.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

from pymatgen.core import Composition, Structure
from pymatgen.io.pwscf import PWInput

from prepare_qe_jobs import (
    SAFE_JOB_ID_RE,
    _kpoint_grid,
    _resolve_executable,
    _settings_hash,
    _sha256,
    _validate_config,
    _validate_pseudopotentials,
)


WORKFLOW_VERSION = "qe_pbe_convergence_sweep_v1"
PROTOCOL_SCHEMA = "qe_convergence_protocol_v1"
PREFLIGHT_NAME = "convergence_preflight.json"
QUEUE_NAME = "convergence_queue_manifest.csv"
MAX_REPRESENTATIVES = 8


def _load_json(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    preferred = [
        "rank", "point_id", "sweep_axis", "level_index",
        "cutoff_level_index", "kpoint_level_index", "representative_id",
        "entry_id", "entry_role", "source_id", "candidate_id", "formula",
        "num_atoms", "cutoff_pair_multiplier",
        "requested_kpoint_spacing_inv_angstrom", "kpoints_grid",
        "ecutwfc_ry", "ecutrho_ry", "source_cif", "source_cif_sha256",
        "copied_cif", "copied_cif_sha256", "qe_input", "qe_input_sha256",
        "qe_output", "run_record", "convergence_settings_hash",
        "job_status", "blockers", "job_record",
    ]
    extras = sorted({key for row in rows for key in row} - set(preferred))
    fields = preferred + extras
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Invalid {label}: booleans are not numeric values")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}: {value!r}")
    return result


def _strict_sequence(
    values: Any, *, label: str, increasing: bool, minimum: float,
) -> list[float]:
    if not isinstance(values, list) or not (3 <= len(values) <= 12):
        raise ValueError(f"{label} must contain 3..12 values")
    result = [_finite_float(value, label) for value in values]
    if any(value < minimum for value in result):
        raise ValueError(f"{label} contains a value below {minimum}")
    pairs = zip(result, result[1:])
    valid = all(left < right for left, right in pairs) if increasing else all(
        left > right for left, right in pairs
    )
    if not valid:
        direction = "increasing" if increasing else "decreasing"
        raise ValueError(f"{label} must be strictly {direction}")
    return result


def validate_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError(f"Expected protocol schema {PROTOCOL_SCHEMA}")
    if protocol.get("target_profile") != "finite_smearing_static_scf":
        raise ValueError("Unsupported convergence target_profile")
    multipliers = _strict_sequence(
        protocol.get("cutoff_pair_multipliers"),
        label="cutoff_pair_multipliers",
        increasing=True,
        minimum=1.0,
    )
    spacings = _strict_sequence(
        protocol.get("kpoint_spacings_inv_angstrom"),
        label="kpoint_spacings_inv_angstrom",
        increasing=False,
        minimum=1e-6,
    )
    dense_spacing = _finite_float(
        protocol.get("cutoff_reference_kpoint_spacing_inv_angstrom"),
        "cutoff_reference_kpoint_spacing_inv_angstrom",
    )
    high_multiplier = _finite_float(
        protocol.get("kpoint_reference_cutoff_multiplier"),
        "kpoint_reference_cutoff_multiplier",
    )
    if not math.isclose(dense_spacing, spacings[-1], rel_tol=0, abs_tol=1e-12):
        raise ValueError("Cutoff sweep must use the densest listed k-point spacing")
    if not math.isclose(high_multiplier, multipliers[-1], rel_tol=0, abs_tol=1e-12):
        raise ValueError("K-point sweep must use the highest listed cutoff multiplier")
    tolerance_caps = {
        "energy_tolerance_mev_per_atom": 20.0,
        "force_component_tolerance_ev_per_angstrom": 1.0,
        "stress_component_tolerance_kbar": 50.0,
    }
    for field, safety_cap in tolerance_caps.items():
        value = _finite_float(protocol.get(field), field)
        if value <= 0:
            raise ValueError(f"{field} must be positive")
        if value > safety_cap:
            raise ValueError(f"{field} exceeds protocol safety cap {safety_cap:g}")
    energy_tolerance = float(protocol["energy_tolerance_mev_per_atom"])
    raw_tail = protocol.get("min_stable_tail_points")
    if isinstance(raw_tail, bool):
        raise ValueError("min_stable_tail_points must be an integer")
    try:
        min_tail = int(raw_tail)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("min_stable_tail_points must be an integer") from exc
    if isinstance(raw_tail, float) and not raw_tail.is_integer():
        raise ValueError("min_stable_tail_points must be an integer")
    if isinstance(raw_tail, str) and str(min_tail) != raw_tail.strip():
        raise ValueError("min_stable_tail_points must be an integer")
    if min_tail < 2 or min_tail > min(len(multipliers), len(spacings)):
        raise ValueError("min_stable_tail_points is outside the tested window")
    return {
        **protocol,
        "cutoff_pair_multipliers": multipliers,
        "kpoint_spacings_inv_angstrom": spacings,
        "cutoff_reference_kpoint_spacing_inv_angstrom": dense_spacing,
        "kpoint_reference_cutoff_multiplier": high_multiplier,
        "energy_tolerance_mev_per_atom": energy_tolerance,
        "force_component_tolerance_ev_per_angstrom": float(
            protocol["force_component_tolerance_ev_per_angstrom"]
        ),
        "stress_component_tolerance_kbar": float(
            protocol["stress_component_tolerance_kbar"]
        ),
        "min_stable_tail_points": min_tail,
    }


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def convergence_settings_hash(payload: dict[str, Any]) -> str:
    return _canonical_digest(payload)


def _validate_static_config(config: dict[str, Any]) -> None:
    _validate_config(config)
    for field in ("static_degauss_ry", "static_conv_thr"):
        value = _finite_float(config.get(field), field)
        if value <= 0:
            raise ValueError(f"Config field must be positive: {field}")
    raw_maxstep = config.get("electron_maxstep")
    if isinstance(raw_maxstep, bool):
        raise ValueError("electron_maxstep must be a positive integer")
    try:
        maxstep = int(raw_maxstep)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("electron_maxstep must be a positive integer") from exc
    if maxstep < 1 or str(maxstep) != str(raw_maxstep).strip():
        raise ValueError("electron_maxstep must be a positive integer")
    mixing_beta = _finite_float(config.get("mixing_beta"), "mixing_beta")
    if not 0 < mixing_beta <= 1:
        raise ValueError("mixing_beta must be in (0, 1]")
    if not str(config.get("diagonalization") or "").strip():
        raise ValueError("diagonalization must be non-empty")


def _point_id(representative_id: str, label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", representative_id)[:40]
    digest = hashlib.sha256(
        f"{representative_id}:{label}".encode("utf-8")
    ).hexdigest()[:10]
    return f"cv_{slug}_{label}_{digest}"


def _verify_source_preflight(
    source_preflight_path: Path,
) -> tuple[dict[str, Any], list[dict[str, str]], Path, Path, Path]:
    source_preflight_path = Path(source_preflight_path).resolve()
    preflight = _load_json(source_preflight_path)
    if preflight.get("status") not in {
        "runnable_not_started", "inputs_ready_engine_missing",
    }:
        raise ValueError("Source candidate preflight has no verified QE inputs")
    if preflight.get("pseudo_blockers"):
        raise ValueError("Source candidate preflight has pseudopotential blockers")
    queue_path = Path(str(preflight.get("queue_manifest") or "")).resolve()
    queue_sha = str(preflight.get("queue_manifest_sha256") or "").lower()
    if not queue_path.is_file() or _sha256(queue_path) != queue_sha:
        raise ValueError("Source candidate queue hash mismatch")
    config_path = Path(str(preflight.get("config") or "")).resolve()
    pseudo_manifest_path = Path(str(preflight.get("pseudo_manifest") or "")).resolve()
    if not config_path.is_file() or not pseudo_manifest_path.is_file():
        raise ValueError("Source config/pseudopotential manifest is missing")
    config = _load_json(config_path)
    _validate_static_config(config)
    base_wfc = _finite_float(preflight.get("global_ecutwfc_ry"), "global_ecutwfc_ry")
    base_rho = _finite_float(preflight.get("global_ecutrho_ry"), "global_ecutrho_ry")
    expected_settings = _settings_hash(
        config_path=config_path,
        pseudo_manifest_path=pseudo_manifest_path,
        global_ecutwfc_ry=base_wfc,
        global_ecutrho_ry=base_rho,
    )
    if expected_settings != preflight.get("relax_input_settings_hash"):
        raise ValueError("Source candidate preflight settings hash mismatch")
    return preflight, _read_csv(queue_path), queue_path, config_path, pseudo_manifest_path


def prepare_convergence_jobs(
    *,
    source_preflight_path: Path,
    representative_ids: list[str],
    protocol_path: Path,
    output_dir: Path,
    pw_executable: str = "pw.x",
    diagnostic_allow_partial_element_coverage: bool = False,
) -> dict[str, Any]:
    if not representative_ids:
        raise ValueError("At least one explicit --candidate-id is required")
    if len(representative_ids) != len(set(representative_ids)):
        raise ValueError("Duplicate representative candidate IDs are not allowed")
    if len(representative_ids) > MAX_REPRESENTATIVES:
        raise ValueError(f"At most {MAX_REPRESENTATIVES} representatives are allowed")
    for identifier in representative_ids:
        if not SAFE_JOB_ID_RE.fullmatch(identifier):
            raise ValueError(f"Unsafe representative candidate ID: {identifier!r}")

    (
        source_preflight,
        source_queue,
        source_queue_path,
        source_config_path,
        source_pseudo_manifest_path,
    ) = _verify_source_preflight(source_preflight_path)
    protocol_source_path = Path(protocol_path).resolve()
    protocol = validate_protocol(_load_json(protocol_source_path))
    source_candidate_ids = [row.get("candidate_id", "") for row in source_queue]
    if (
        any(not identifier for identifier in source_candidate_ids)
        or len(source_candidate_ids) != len(set(source_candidate_ids))
    ):
        raise ValueError("Source queue candidate IDs are missing or duplicated")
    by_candidate = {row["candidate_id"]: row for row in source_queue}
    missing = set(representative_ids) - set(by_candidate)
    if missing:
        raise ValueError(f"Representative IDs absent from source queue: {sorted(missing)}")

    required_elements = set(source_preflight.get("required_elements") or [])
    if not required_elements:
        raise ValueError("Source preflight has no required element inventory")
    source_structures: dict[str, Structure] = {}
    computed_required_elements: set[str] = set()
    for row in source_queue:
        identifier = row["candidate_id"]
        source_cif = Path(str(row.get("source_cif") or "")).resolve()
        copied_cif = Path(str(row.get("copied_cif") or "")).resolve()
        expected_sha = str(row.get("source_cif_sha256") or "").lower()
        if (
            not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            or not source_cif.is_file()
            or _sha256(source_cif) != expected_sha
            or not copied_cif.is_file()
            or _sha256(copied_cif) != expected_sha
        ):
            raise ValueError(f"Source queue CIF hash mismatch: {identifier}")
        structure = Structure.from_file(copied_cif)
        expected_formula = Composition(row["formula"]).reduced_formula
        if structure.composition.reduced_formula != expected_formula:
            raise ValueError(f"Source queue formula mismatch: {identifier}")
        source_structures[identifier] = structure
        computed_required_elements.update(
            element.symbol for element in structure.composition.elements
        )
    if computed_required_elements != required_elements:
        raise ValueError(
            "Source preflight element inventory differs from its complete queue"
        )
    pseudo_dir = Path(source_preflight_path).resolve().parent / "pseudos"
    pseudo_entries, pseudo_blockers, pseudo_manifest = _validate_pseudopotentials(
        required_elements=sorted(required_elements),
        pseudo_manifest_path=source_pseudo_manifest_path,
        pseudo_dir=pseudo_dir,
    )
    if pseudo_blockers or pseudo_manifest is None:
        raise ValueError(
            "Source pseudopotentials failed convergence re-verification: "
            + "; ".join(pseudo_blockers)
        )
    base_wfc = _finite_float(source_preflight["global_ecutwfc_ry"], "global_ecutwfc_ry")
    base_rho = _finite_float(source_preflight["global_ecutrho_ry"], "global_ecutrho_ry")
    recommended_wfc = max(float(entry["ecutwfc_ry"]) for entry in pseudo_entries.values())
    recommended_rho = max(float(entry["ecutrho_ry"]) for entry in pseudo_entries.values())
    if base_wfc < recommended_wfc or base_rho < recommended_rho:
        raise ValueError("Source base cutoffs are below verified SSSP recommendations")
    if base_rho < base_wfc:
        raise ValueError("Source ecutrho must not be lower than ecutwfc")

    output_dir = Path(output_dir).resolve()
    if (output_dir / PREFLIGHT_NAME).exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing convergence preflight: {output_dir}"
        )
    jobs_dir = output_dir / "jobs"
    representatives_dir = output_dir / "representatives"
    bundled_pseudo_dir = output_dir / "pseudos"
    for directory in (jobs_dir, representatives_dir, bundled_pseudo_dir):
        directory.mkdir(parents=True, exist_ok=True)
    config_snapshot = output_dir / "workflow_config_snapshot.json"
    pseudo_manifest_snapshot = output_dir / "pseudo_manifest_snapshot.json"
    protocol_snapshot = output_dir / "convergence_protocol_snapshot.json"
    shutil.copy2(source_config_path, config_snapshot)
    shutil.copy2(source_pseudo_manifest_path, pseudo_manifest_snapshot)
    shutil.copy2(protocol_source_path, protocol_snapshot)
    for entry in pseudo_entries.values():
        shutil.copy2(entry["source_path"], bundled_pseudo_dir / entry["filename"])

    config = _load_json(config_snapshot)
    _validate_static_config(config)
    selected_rows: list[tuple[dict[str, str], Structure, Path, Path]] = []
    representative_records: list[dict[str, Any]] = []
    covered_elements: set[str] = set()
    for identifier in representative_ids:
        row = by_candidate[identifier]
        copied_source = Path(str(row.get("copied_cif") or "")).resolve()
        original_source = Path(str(row.get("source_cif") or "")).resolve()
        expected_source_sha = str(row.get("source_cif_sha256") or "").lower()
        if (
            not re.fullmatch(r"[0-9a-f]{64}", expected_source_sha)
            or not original_source.is_file()
            or _sha256(original_source) != expected_source_sha
            or not copied_source.is_file()
            or _sha256(copied_source) != expected_source_sha
        ):
            raise ValueError(f"Representative CIF hash mismatch: {identifier}")
        structure = source_structures[identifier]
        expected_formula = Composition(row["formula"]).reduced_formula
        if structure.composition.reduced_formula != expected_formula:
            raise ValueError(f"Representative formula mismatch: {identifier}")
        elements = {element.symbol for element in structure.composition.elements}
        covered_elements.update(elements)
        representative_copy = representatives_dir / f"{identifier}.cif"
        shutil.copy2(copied_source, representative_copy)
        selected_rows.append((row, structure, copied_source, representative_copy))
        representative_records.append({
            "representative_id": identifier,
            "source_id": row.get("source_id") or identifier,
            "formula": expected_formula,
            "elements": sorted(elements),
            "num_atoms": float(structure.composition.num_atoms),
            "source_cif": str(copied_source),
            "source_cif_sha256": expected_source_sha,
            "copied_cif": str(representative_copy),
            "copied_cif_sha256": _sha256(representative_copy),
        })
    missing_element_coverage = sorted(required_elements - covered_elements)
    coverage_complete = not missing_element_coverage
    if missing_element_coverage and not diagnostic_allow_partial_element_coverage:
        raise ValueError(
            "Representatives do not cover all candidate-queue elements: "
            + ",".join(missing_element_coverage)
        )

    multipliers = protocol["cutoff_pair_multipliers"]
    spacings = protocol["kpoint_spacings_inv_angstrom"]
    dense_spacing = float(protocol["cutoff_reference_kpoint_spacing_inv_angstrom"])
    high_multiplier = float(protocol["kpoint_reference_cutoff_multiplier"])
    point_specs: list[dict[str, Any]] = []
    deduplicated_requests: list[dict[str, Any]] = []
    rank = 0
    for row, structure, source_cif, representative_copy in selected_rows:
        representative_id = row["candidate_id"]
        dense_grid = _kpoint_grid(structure, dense_spacing)
        for cutoff_index, multiplier in enumerate(multipliers):
            rank += 1
            is_anchor = cutoff_index == len(multipliers) - 1
            label = "anchor" if is_anchor else f"cutoff_{cutoff_index:02d}"
            point_specs.append({
                "rank": rank,
                "point_id": _point_id(representative_id, label),
                "sweep_axis": "anchor" if is_anchor else "cutoff_pair",
                "level_index": cutoff_index,
                "cutoff_level_index": cutoff_index,
                "kpoint_level_index": len(spacings) - 1,
                "representative_id": representative_id,
                "source_id": row.get("source_id") or representative_id,
                "formula": structure.composition.reduced_formula,
                "num_atoms": float(structure.composition.num_atoms),
                "source_cif": str(source_cif),
                "source_cif_sha256": _sha256(source_cif),
                "copied_cif": str(representative_copy),
                "copied_cif_sha256": _sha256(representative_copy),
                "cutoff_pair_multiplier": multiplier,
                "requested_kpoint_spacing_inv_angstrom": dense_spacing,
                "kpoints_grid": list(dense_grid),
                "ecutwfc_ry": base_wfc * multiplier,
                "ecutrho_ry": base_rho * multiplier,
            })
        seen_grids = {dense_grid}
        for kpoint_index, spacing in enumerate(spacings[:-1]):
            grid = _kpoint_grid(structure, float(spacing))
            if grid in seen_grids:
                deduplicated_requests.append({
                    "representative_id": representative_id,
                    "requested_spacing_inv_angstrom": spacing,
                    "duplicate_grid": list(grid),
                })
                continue
            seen_grids.add(grid)
            rank += 1
            point_specs.append({
                "rank": rank,
                "point_id": _point_id(representative_id, f"kpoint_{kpoint_index:02d}"),
                "sweep_axis": "kpoint",
                "level_index": kpoint_index,
                "cutoff_level_index": len(multipliers) - 1,
                "kpoint_level_index": kpoint_index,
                "representative_id": representative_id,
                "source_id": row.get("source_id") or representative_id,
                "formula": structure.composition.reduced_formula,
                "num_atoms": float(structure.composition.num_atoms),
                "source_cif": str(source_cif),
                "source_cif_sha256": _sha256(source_cif),
                "copied_cif": str(representative_copy),
                "copied_cif_sha256": _sha256(representative_copy),
                "cutoff_pair_multiplier": high_multiplier,
                "requested_kpoint_spacing_inv_angstrom": spacing,
                "kpoints_grid": list(grid),
                "ecutwfc_ry": base_wfc * high_multiplier,
                "ecutrho_ry": base_rho * high_multiplier,
            })
        unique_kpoint_grids = {
            tuple(point["kpoints_grid"])
            for point in point_specs
            if point["representative_id"] == representative_id
            and point["sweep_axis"] in {"kpoint", "anchor"}
        }
        if len(unique_kpoint_grids) < 3:
            raise ValueError(
                f"Representative {representative_id} yields fewer than three "
                "distinct k-point grids; extend the protocol spacing range"
            )

    settings_payload = {
        "workflow_version": WORKFLOW_VERSION,
        "source_preflight_sha256": _sha256(Path(source_preflight_path).resolve()),
        "source_queue_sha256": _sha256(source_queue_path),
        "config_sha256": _sha256(config_snapshot),
        "pseudo_manifest_sha256": _sha256(pseudo_manifest_snapshot),
        "protocol_sha256": _sha256(protocol_snapshot),
        "protocol_values": protocol,
        "base_ecutwfc_ry": base_wfc,
        "base_ecutrho_ry": base_rho,
        "representatives": representative_records,
        "required_elements": sorted(required_elements),
        "covered_elements": sorted(covered_elements),
        "coverage_complete": coverage_complete,
        "pseudopotentials": [
            {
                "element": symbol,
                "filename": pseudo_entries[symbol]["filename"],
                "sha256": pseudo_entries[symbol]["sha256"],
            }
            for symbol in sorted(required_elements)
        ],
        "study_points": [
            {
                key: point[key]
                for key in (
                    "rank", "point_id", "sweep_axis", "level_index",
                    "cutoff_level_index", "kpoint_level_index",
                    "representative_id", "source_id", "formula", "num_atoms",
                    "cutoff_pair_multiplier",
                    "requested_kpoint_spacing_inv_angstrom", "kpoints_grid",
                    "ecutwfc_ry", "ecutrho_ry", "source_cif_sha256",
                    "copied_cif_sha256",
                )
            }
            for point in point_specs
        ],
    }
    settings_hash = convergence_settings_hash(settings_payload)
    executable_path = _resolve_executable(pw_executable)
    engine_blockers = [] if executable_path else [f"pw_executable_not_found:{pw_executable}"]
    pseudo_map_by_rep = {
        rep["representative_id"]: {
            symbol: pseudo_entries[symbol]["filename"]
            for symbol in rep["elements"]
        }
        for rep in representative_records
    }
    structure_by_rep = {
        row["candidate_id"]: structure
        for row, structure, _source, _copy in selected_rows
    }
    queue_rows: list[dict[str, Any]] = []
    for point in point_specs:
        point_id = point["point_id"]
        job_dir = jobs_dir / f"{int(point['rank']):03d}_{point_id}"
        job_dir.mkdir(parents=True, exist_ok=True)
        input_path = job_dir / "convergence-scf.in"
        output_path = job_dir / "convergence-scf.out"
        run_record = job_dir / "convergence_run.json"
        structure = structure_by_rep[point["representative_id"]]
        pw_input = PWInput(
            structure,
            pseudo=pseudo_map_by_rep[point["representative_id"]],
            control={
                "calculation": "scf",
                "restart_mode": "from_scratch",
                "prefix": point_id,
                "pseudo_dir": "../../pseudos",
                "outdir": "./tmp",
                "disk_io": "low",
                "tstress": True,
                "tprnfor": True,
            },
            system={
                "input_dft": config["input_dft"],
                "ecutwfc": point["ecutwfc_ry"],
                "ecutrho": point["ecutrho_ry"],
                "occupations": config["occupations"],
                "smearing": config["smearing"],
                "degauss": float(config["static_degauss_ry"]),
            },
            electrons={
                "conv_thr": float(config["static_conv_thr"]),
                "electron_maxstep": int(config["electron_maxstep"]),
                "mixing_beta": float(config["mixing_beta"]),
                "diagonalization": config["diagonalization"],
            },
            kpoints_mode="automatic",
            kpoints_grid=tuple(point["kpoints_grid"]),
            kpoints_shift=(0, 0, 0),
        )
        pw_input.write_file(input_path)
        # pymatgen emits empty &IONS and &CELL namelists even for SCF inputs.
        # Remove only those exactly empty sections so the fixed-geometry intent
        # is explicit and can be verified without relying on QE ignoring them.
        input_text = input_path.read_text(encoding="utf-8")
        input_text = re.sub(
            r"(?im)^\s*&(IONS|CELL)\s*\n\s*/\s*\n", "", input_text
        )
        input_path.write_text(input_text, encoding="utf-8")
        blockers = list(engine_blockers)
        job_status = "runnable_not_started" if executable_path else "inputs_ready_engine_missing"
        entry_id = f"convergence:{point_id}"
        job_record = {
            "workflow_version": WORKFLOW_VERSION,
            **point,
            "entry_id": entry_id,
            "entry_role": "convergence",
            "candidate_id": point_id,
            "job_status": job_status,
            "blockers": blockers,
            "qe_input": str(input_path),
            "qe_input_sha256": _sha256(input_path),
            "qe_output": str(output_path),
            "run_record": str(run_record),
            "convergence_settings_hash": settings_hash,
            "calculation_started": False,
        }
        job_record_path = job_dir / "job_plan.json"
        job_record_path.write_text(json.dumps(job_record, indent=2), encoding="utf-8")
        queue_rows.append({
            **job_record,
            "kpoints_grid": "x".join(str(value) for value in point["kpoints_grid"]),
            "blockers": json.dumps(blockers),
            "job_record": str(job_record_path),
        })

    queue_path = output_dir / QUEUE_NAME
    _write_csv(queue_path, queue_rows)
    preflight = {
        "workflow_version": WORKFLOW_VERSION,
        "status": "runnable_not_started" if executable_path else "inputs_ready_engine_missing",
        "calculation": "scf",
        "stage": "sweep",
        "calculation_started": False,
        "source_preflight": str(Path(source_preflight_path).resolve()),
        "source_preflight_sha256": _sha256(Path(source_preflight_path).resolve()),
        "source_queue_manifest": str(source_queue_path),
        "source_queue_manifest_sha256": _sha256(source_queue_path),
        "config": str(config_snapshot),
        "pseudo_manifest": str(pseudo_manifest_snapshot),
        "protocol": str(protocol_snapshot),
        "protocol_sha256": _sha256(protocol_snapshot),
        "protocol_values": protocol,
        "base_ecutwfc_ry": base_wfc,
        "base_ecutrho_ry": base_rho,
        "required_elements": sorted(required_elements),
        "covered_elements": sorted(covered_elements),
        "missing_element_coverage": missing_element_coverage,
        "coverage_complete": coverage_complete,
        "confirmation_eligible": (
            coverage_complete and not diagnostic_allow_partial_element_coverage
        ),
        "certificate_eligible": False,
        "diagnostic_allow_partial_element_coverage": diagnostic_allow_partial_element_coverage,
        "representatives": representative_records,
        "study_points": settings_payload["study_points"],
        "deduplicated_kpoint_requests": deduplicated_requests,
        "convergence_settings_payload": settings_payload,
        "convergence_settings_hash": settings_hash,
        "bundled_pseudopotentials": [
            {
                "element": symbol,
                "filename": pseudo_entries[symbol]["filename"],
                "path": str(bundled_pseudo_dir / pseudo_entries[symbol]["filename"]),
                "sha256": pseudo_entries[symbol]["sha256"],
            }
            for symbol in sorted(required_elements)
        ],
        "pw_executable_requested": pw_executable,
        "pw_executable_resolved": executable_path,
        "engine_blockers": engine_blockers,
        "resource_policy": source_preflight.get("resource_policy") or {},
        "queue_manifest": str(queue_path),
        "queue_manifest_sha256": _sha256(queue_path),
        "scientific_limit": (
            "Prepared sparse static-SCF convergence sweep only. A provisional "
            "selection still requires completed outputs and an independent "
            "confirmation run before production use."
        ),
    }
    preflight_path = output_dir / PREFLIGHT_NAME
    preflight_path.write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    print(f"Convergence preparation status: {preflight['status']}")
    print(f"Representatives: {len(representative_records)}")
    print(f"SCF points: {len(queue_rows)}")
    print(f"Queue:     {queue_path}")
    print(f"Preflight: {preflight_path}")
    print("No Quantum ESPRESSO calculation was started.")
    return preflight


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-preflight", type=Path, required=True)
    parser.add_argument("--candidate-id", action="append", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pw-executable", default="pw.x")
    parser.add_argument(
        "--diagnostic-allow-partial-element-coverage", action="store_true"
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    prepare_convergence_jobs(
        source_preflight_path=args.source_preflight,
        representative_ids=args.candidate_id,
        protocol_path=args.protocol,
        output_dir=args.output_dir,
        pw_executable=args.pw_executable,
        diagnostic_allow_partial_element_coverage=(
            args.diagnostic_allow_partial_element_coverage
        ),
    )
