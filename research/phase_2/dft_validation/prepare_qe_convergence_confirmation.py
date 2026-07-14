"""Prepare independent static-SCF confirmation jobs for a QE convergence study.

This command consumes a completed, provenance-verified provisional convergence
summary and its original sweep preflight.  It prepares exactly one fixed-
geometry SCF calculation per representative at the provisional *global*
cutoff/k-point combination.  That combined point is intentionally separate
from the two one-dimensional sweep axes and tests whether their provisional
selections remain converged when applied together.

The module is plan-only: it writes inputs and locked manifests but never starts
Quantum ESPRESSO.  Execute the resulting queue later with ``run_qe_jobs.py``.
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

from collect_qe_convergence import (
    COLLECTOR_VERSION,
    HASH_RE,
    _bundled_pseudo_blockers,
    _verify_preflight,
)
from prepare_qe_convergence_jobs import (
    _finite_float,
    _validate_static_config,
    convergence_settings_hash,
)
from prepare_qe_jobs import (
    SAFE_JOB_ID_RE,
    _ensure_fresh_output_dir,
    _kpoint_grid,
    _qe_scratch_outdir,
    _qe_scratch_campaign_namespace,
    _prepare_qe_scratch_campaign,
    _resolve_executable,
    _resolve_scratch_root,
    _sha256,
)
from qe_execution_provenance import (
    require_same_execution_provenance,
    validate_execution_provenance,
)
from qe_convergence_evidence import verify_provisional_sweep_evidence


WORKFLOW_VERSION = "qe_pbe_convergence_confirmation_v1"
PREFLIGHT_NAME = "confirmation_preflight.json"
QUEUE_NAME = "confirmation_queue_manifest.csv"
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")


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
        "rank", "point_id", "representative_id", "entry_id", "entry_role",
        "source_id", "candidate_id", "formula", "num_atoms",
        "sweep_anchor_point_id", "selected_cutoff_source_point_id",
        "selected_kpoint_source_point_id", "cutoff_pair_multiplier",
        "requested_kpoint_spacing_inv_angstrom", "kpoints_grid",
        "ecutwfc_ry", "ecutrho_ry", "source_cif", "source_cif_sha256",
        "copied_cif", "copied_cif_sha256", "qe_input", "qe_input_sha256",
        "qe_output", "run_record", "qe_scratch_outdir",
        "convergence_settings_hash", "job_status",
        "blockers", "job_record",
    ]
    extras = sorted({key for row in rows for key in row} - set(preferred))
    fields = preferred + extras
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def _close(left: Any, right: Any, *, label: str) -> float:
    first = _finite_float(left, label)
    second = _finite_float(right, label)
    if not math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-10):
        raise ValueError(f"Provisional convergence selection mismatch: {label}")
    return first


def _confirmation_point_id(representative_id: str, settings_hash_seed: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", representative_id)[:40]
    digest = hashlib.sha256(
        f"{representative_id}:{settings_hash_seed}".encode("utf-8")
    ).hexdigest()[:10]
    return f"cf_{slug}_{digest}"


def _component_array(value: Any, *, label: str, rows: int) -> list[list[float]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid {label} JSON") from exc
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"Invalid {label} row count")
    result: list[list[float]] = []
    for vector in value:
        if not isinstance(vector, list) or len(vector) != 3:
            raise ValueError(f"Invalid {label} component shape")
        result.append([
            _finite_float(component, label) for component in vector
        ])
    return result


def _verify_provisional_summary(
    *, summary_path: Path, sweep_preflight_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, str, str, Path, str]:
    """Fail closed unless summary and sweep encode one consistent selection."""

    summary_path = Path(summary_path).resolve()
    sweep_preflight_path = Path(sweep_preflight_path).resolve()
    sweep, sweep_preflight_sha, sweep_settings_hash = _verify_preflight(
        sweep_preflight_path
    )
    if _bundled_pseudo_blockers(sweep):
        raise ValueError(
            "Sweep pseudopotential bundle no longer matches its locked inventory"
        )
    if sweep.get("confirmation_eligible") is not True:
        raise ValueError("Sweep preflight is not eligible for confirmation")

    summary_sha = _sha256(summary_path)
    summary = _load_json(summary_path)
    required_summary_state = {
        "collector_version": COLLECTOR_VERSION,
        "status": "provisional_selection_ready",
        "confirmation_eligible": True,
        "confirmation_required": True,
        "certificate_eligible": False,
        "cutoff_window_converged": True,
        "kpoint_window_converged": True,
        "coverage_complete": True,
    }
    for key, expected in required_summary_state.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"Provisional summary is not confirmation-ready: {key}"
            )
    if summary.get("global_failures") != []:
        raise ValueError("Provisional summary contains global convergence failures")
    if summary.get("convergence_settings_hash") != sweep_settings_hash:
        raise ValueError("Summary and sweep convergence settings hashes differ")
    if Path(str(summary.get("convergence_preflight") or "")).resolve() != sweep_preflight_path:
        raise ValueError("Summary references a different convergence preflight")
    if summary.get("convergence_preflight_sha256") != sweep_preflight_sha:
        raise ValueError("Summary convergence preflight hash mismatch")
    if summary.get("candidate_source_preflight") != sweep.get("source_preflight"):
        raise ValueError("Summary candidate source preflight mismatch")
    if summary.get("candidate_source_preflight_sha256") != sweep.get(
        "source_preflight_sha256"
    ):
        raise ValueError("Summary candidate source preflight hash mismatch")

    results_path = Path(str(summary.get("results") or "")).resolve()
    results_sha = str(summary.get("results_sha256") or "").lower()
    if (
        not results_path.is_file()
        or not HASH_PATTERN.fullmatch(results_sha)
        or _sha256(results_path) != results_sha
    ):
        raise ValueError("Provisional convergence results hash mismatch")
    result_rows = _read_csv(results_path)
    points = summary.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("Provisional summary has no point inventory")
    if summary.get("point_count") != len(points) or len(result_rows) != len(points):
        raise ValueError("Provisional result point count mismatch")
    if summary.get("candidate_count") != len(sweep.get("representatives") or []):
        raise ValueError("Provisional representative count mismatch")
    representatives = sweep.get("representatives")
    if not isinstance(representatives, list) or not representatives:
        raise ValueError("Sweep representative inventory is missing")
    representative_ids = [
        str(rep.get("representative_id") or "") for rep in representatives
    ]
    evidence = verify_provisional_sweep_evidence(
        summary=summary,
        csv_rows=result_rows,
        locked_study_points=sweep.get("study_points") or [],
        representative_ids=representative_ids,
        expected_protocol=sweep.get("protocol_values"),
    )
    point_by_id: dict[str, dict[str, Any]] = evidence["point_by_id"]
    # Keep the explicit locked-study identity checks below as a second, local
    # guard around fields used to build PWInput files.
    sweep_point_by_id = {
        point["point_id"]: point for point in sweep.get("study_points") or []
    }
    if (
        len(sweep_point_by_id) != len(sweep.get("study_points") or [])
        or set(sweep_point_by_id) != set(point_by_id)
    ):
        raise ValueError("Provisional points differ from the locked sweep study")
    for point_id, point in point_by_id.items():
        canonical = sweep_point_by_id[point_id]
        for key in (
            "representative_id", "sweep_axis", "cutoff_level_index",
            "kpoint_level_index", "cutoff_pair_multiplier",
            "requested_kpoint_spacing_inv_angstrom", "ecutwfc_ry", "ecutrho_ry",
        ):
            if str(point.get(key)) != str(canonical.get(key)):
                raise ValueError(f"Provisional point differs from sweep: {point_id}:{key}")
        expected_grid = "x".join(str(value) for value in canonical["kpoints_grid"])
        if point.get("kpoints_grid") != expected_grid:
            raise ValueError(f"Provisional point k-grid differs from sweep: {point_id}")
    result_ids = [row.get("point_id", "") for row in result_rows]
    if set(result_ids) != set(point_by_id) or len(result_ids) != len(set(result_ids)):
        raise ValueError("Provisional CSV and summary point inventories differ")
    for row in result_rows:
        point = point_by_id[row["point_id"]]
        for key in (
            "representative_id", "sweep_axis", "qe_input_sha256",
            "qe_output_sha256", "qe_program_version", "pw_executable_sha256",
            "execution_provenance",
        ):
            if str(row.get(key) or "") != str(point.get(key) or ""):
                raise ValueError(f"Provisional CSV point mismatch: {row['point_id']}:{key}")

    program_version = str(summary.get("qe_program_version") or "").strip()
    executable_sha = str(summary.get("pw_executable_sha256") or "").lower()
    if not program_version or not HASH_RE.fullmatch(executable_sha):
        raise ValueError("Provisional QE executable provenance is incomplete")
    if any(
        point.get("qe_program_version") != program_version
        or point.get("pw_executable_sha256") != executable_sha
        for point in points
    ):
        raise ValueError("Provisional points have mixed QE executable provenance")
    baseline_execution = validate_execution_provenance(
        summary.get("execution_provenance")
    )
    for point in points:
        try:
            point_execution = json.loads(
                str(point.get("execution_provenance") or "")
            )
            require_same_execution_provenance(
                point_execution, baseline_execution, label="provisional sweep",
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                "Provisional points have mixed or invalid execution provenance"
            ) from exc

    if (
        any(not SAFE_JOB_ID_RE.fullmatch(value) for value in representative_ids)
        or len(representative_ids) != len(set(representative_ids))
    ):
        raise ValueError("Sweep representative IDs are invalid or duplicated")
    selections = summary.get("selections_by_representative")
    if not isinstance(selections, dict) or set(selections) != set(representative_ids):
        raise ValueError("Provisional selections do not cover every representative")

    selected_cutoffs: list[dict[str, Any]] = []
    selected_kpoints: list[dict[str, Any]] = []
    for representative_id in representative_ids:
        selection = selections[representative_id]
        if not isinstance(selection, dict):
            raise ValueError(f"Invalid provisional selection: {representative_id}")
        cutoff = selection.get("selected_cutoff_pair")
        kpoint = selection.get("selected_kpoint")
        if not isinstance(cutoff, dict) or not isinstance(kpoint, dict):
            raise ValueError(f"Incomplete provisional selection: {representative_id}")
        cutoff_point = point_by_id.get(str(cutoff.get("point_id") or ""))
        kpoint_point = point_by_id.get(str(kpoint.get("point_id") or ""))
        anchor_point = point_by_id.get(str(selection.get("anchor_point_id") or ""))
        if any(point is None for point in (cutoff_point, kpoint_point, anchor_point)):
            raise ValueError(f"Provisional source point is missing: {representative_id}")
        assert cutoff_point is not None and kpoint_point is not None and anchor_point is not None
        if any(
            point.get("representative_id") != representative_id
            for point in (cutoff_point, kpoint_point, anchor_point)
        ) or anchor_point.get("sweep_axis") != "anchor":
            raise ValueError(f"Provisional source point identity mismatch: {representative_id}")
        if cutoff_point.get("sweep_axis") not in {"cutoff_pair", "anchor"}:
            raise ValueError(f"Selected cutoff point is on the wrong axis: {representative_id}")
        if kpoint_point.get("sweep_axis") not in {"kpoint", "anchor"}:
            raise ValueError(f"Selected k-point is on the wrong axis: {representative_id}")
        if any(
            point.get("point_within_tolerances") is not True
            or point.get("stable_tail_from_this_level") is not True
            for point in (cutoff_point, kpoint_point)
        ):
            raise ValueError(f"Selected point lacks a passing stable tail: {representative_id}")
        _close(cutoff["multiplier"], cutoff_point["cutoff_pair_multiplier"], label="cutoff multiplier")
        _close(cutoff["ecutwfc_ry"], cutoff_point["ecutwfc_ry"], label="ecutwfc")
        _close(cutoff["ecutrho_ry"], cutoff_point["ecutrho_ry"], label="ecutrho")
        _close(
            kpoint["spacing_inv_angstrom"],
            kpoint_point["requested_kpoint_spacing_inv_angstrom"],
            label="k-point spacing",
        )
        if str(kpoint.get("kpoints_grid")) != str(kpoint_point.get("kpoints_grid")):
            raise ValueError(f"Provisional selected k-grid mismatch: {representative_id}")
        selected_cutoffs.append(cutoff)
        selected_kpoints.append(kpoint)

    expected_global = {
        "ecutwfc_ry": max(float(value["ecutwfc_ry"]) for value in selected_cutoffs),
        "ecutrho_ry": max(float(value["ecutrho_ry"]) for value in selected_cutoffs),
        "cutoff_pair_multiplier": max(float(value["multiplier"]) for value in selected_cutoffs),
        "kpoint_spacing_inv_angstrom": min(
            float(value["spacing_inv_angstrom"]) for value in selected_kpoints
        ),
    }
    global_selection = summary.get("provisional_global_selection")
    if not isinstance(global_selection, dict) or set(expected_global) - set(global_selection):
        raise ValueError("Provisional global selection is missing")
    for key, expected in expected_global.items():
        _close(global_selection[key], expected, label=key)
    base_wfc = _finite_float(sweep.get("base_ecutwfc_ry"), "base_ecutwfc_ry")
    base_rho = _finite_float(sweep.get("base_ecutrho_ry"), "base_ecutrho_ry")
    multiplier = _finite_float(global_selection["cutoff_pair_multiplier"], "cutoff_pair_multiplier")
    _close(global_selection["ecutwfc_ry"], base_wfc * multiplier, label="global ecutwfc")
    _close(global_selection["ecutrho_ry"], base_rho * multiplier, label="global ecutrho")
    if _finite_float(global_selection["kpoint_spacing_inv_angstrom"], "kpoint spacing") <= 0:
        raise ValueError("Provisional global k-point spacing must be positive")
    if _sha256(summary_path) != summary_sha:
        raise ValueError("Provisional summary changed during verification")
    return (
        summary, sweep, summary_sha, sweep_preflight_sha, sweep_settings_hash,
        results_path, results_sha,
    )


def prepare_confirmation_jobs(
    *,
    provisional_summary_path: Path,
    sweep_preflight_path: Path,
    output_dir: Path,
    pw_executable: str | None = None,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Write one confirmation SCF input per representative; never execute it."""

    provisional_summary_path = Path(provisional_summary_path).resolve()
    sweep_preflight_path = Path(sweep_preflight_path).resolve()
    (
        summary, sweep, summary_sha, sweep_preflight_sha, sweep_settings_hash,
        results_path, results_sha,
    ) = _verify_provisional_summary(
        summary_path=provisional_summary_path,
        sweep_preflight_path=sweep_preflight_path,
    )
    baseline_execution = validate_execution_provenance(
        summary.get("execution_provenance")
    )

    output_dir = _ensure_fresh_output_dir(output_dir)
    jobs_dir = output_dir / "jobs"
    representatives_dir = output_dir / "representatives"
    bundled_pseudo_dir = output_dir / "pseudos"
    resolved_scratch_root = _resolve_scratch_root(scratch_root)
    scratch_campaign_namespace = _qe_scratch_campaign_namespace(
        workflow_version=WORKFLOW_VERSION, output_dir=output_dir
    )
    scratch_campaign_dir = _prepare_qe_scratch_campaign(
        scratch_root=resolved_scratch_root,
        campaign_namespace=scratch_campaign_namespace,
    )
    for directory in (jobs_dir, representatives_dir, bundled_pseudo_dir):
        directory.mkdir(parents=True, exist_ok=True)

    config_snapshot = output_dir / "workflow_config_snapshot.json"
    pseudo_manifest_snapshot = output_dir / "pseudo_manifest_snapshot.json"
    protocol_snapshot = output_dir / "convergence_protocol_snapshot.json"
    summary_snapshot = output_dir / "provisional_convergence_summary_snapshot.json"
    results_snapshot = output_dir / "provisional_convergence_results_snapshot.csv"
    for source, destination in (
        (Path(sweep["config"]), config_snapshot),
        (Path(sweep["pseudo_manifest"]), pseudo_manifest_snapshot),
        (Path(sweep["protocol"]), protocol_snapshot),
        (provisional_summary_path, summary_snapshot),
        (results_path, results_snapshot),
    ):
        shutil.copy2(source, destination)
    if _sha256(summary_snapshot) != summary_sha:
        raise ValueError("Provisional summary changed while it was being copied")
    if _sha256(results_snapshot) != results_sha:
        raise ValueError("Provisional results changed while they were being copied")
    sweep_payload = sweep["convergence_settings_payload"]
    copied_hash_expectations = (
        (config_snapshot, sweep_payload["config_sha256"], "config"),
        (pseudo_manifest_snapshot, sweep_payload["pseudo_manifest_sha256"], "pseudo manifest"),
        (protocol_snapshot, sweep_payload["protocol_sha256"], "protocol"),
    )
    for path, expected_sha, label in copied_hash_expectations:
        if _sha256(path) != expected_sha:
            raise ValueError(f"Locked sweep {label} changed while it was being copied")

    config = _load_json(config_snapshot)
    _validate_static_config(config)
    required_elements = list(sweep["required_elements"])
    pseudo_by_element: dict[str, dict[str, Any]] = {}
    bundled_pseudopotentials: list[dict[str, Any]] = []
    for entry in sweep["bundled_pseudopotentials"]:
        element = str(entry.get("element") or "")
        filename = str(entry.get("filename") or "")
        expected_sha = str(entry.get("sha256") or "").lower()
        filename_path = Path(filename)
        if (
            not element
            or element in pseudo_by_element
            or not filename
            or filename_path.is_absolute()
            or filename_path.parent != Path(".")
            or not HASH_PATTERN.fullmatch(expected_sha)
        ):
            raise ValueError(f"Invalid bundled pseudopotential inventory entry: {element}")
        source = Path(str(entry.get("path") or "")).resolve()
        if not source.is_file() or _sha256(source) != expected_sha:
            raise ValueError(f"Bundled pseudopotential hash mismatch: {element}")
        destination = bundled_pseudo_dir / filename
        shutil.copy2(source, destination)
        if _sha256(destination) != expected_sha:
            raise ValueError(f"Copied pseudopotential hash mismatch: {element}")
        pseudo_by_element[element] = {"filename": filename, "sha256": expected_sha}
        bundled_pseudopotentials.append({
            "element": element,
            "filename": filename,
            "path": str(destination),
            "sha256": expected_sha,
        })
    if set(pseudo_by_element) != set(required_elements):
        raise ValueError("Confirmation pseudopotential element inventory mismatch")

    selections = summary["selections_by_representative"]
    point_by_id = {point["point_id"]: point for point in summary["points"]}
    selected = summary["provisional_global_selection"]
    cutoff_multiplier = float(selected["cutoff_pair_multiplier"])
    ecutwfc = float(selected["ecutwfc_ry"])
    ecutrho = float(selected["ecutrho_ry"])
    spacing = float(selected["kpoint_spacing_inv_angstrom"])
    representatives: list[dict[str, Any]] = []
    structures: dict[str, Structure] = {}
    study_points: list[dict[str, Any]] = []
    settings_seed = convergence_settings_hash({
        "sweep_settings_hash": sweep_settings_hash,
        "provisional_summary_sha256": summary_sha,
        "selected_global_settings": selected,
    })
    for rank, source_rep in enumerate(sweep["representatives"], start=1):
        representative_id = source_rep["representative_id"]
        source_cif = Path(str(source_rep.get("copied_cif") or "")).resolve()
        source_sha = str(source_rep.get("copied_cif_sha256") or "").lower()
        if not source_cif.is_file() or _sha256(source_cif) != source_sha:
            raise ValueError(f"Sweep representative CIF hash mismatch: {representative_id}")
        structure = Structure.from_file(source_cif)
        formula = structure.composition.reduced_formula
        if formula != Composition(source_rep["formula"]).reduced_formula:
            raise ValueError(f"Sweep representative formula mismatch: {representative_id}")
        elements = sorted(element.symbol for element in structure.composition.elements)
        if elements != sorted(source_rep["elements"]):
            raise ValueError(f"Sweep representative element mismatch: {representative_id}")
        num_atoms = float(structure.composition.num_atoms)
        _close(num_atoms, source_rep["num_atoms"], label="representative num_atoms")
        copied_cif = representatives_dir / f"{representative_id}.cif"
        shutil.copy2(source_cif, copied_cif)
        copied_sha = _sha256(copied_cif)
        if copied_sha != source_sha:
            raise ValueError(f"Copied representative CIF hash mismatch: {representative_id}")
        selection = selections[representative_id]
        anchor = point_by_id[selection["anchor_point_id"]]
        force_reference = _component_array(
            anchor["force_components_ev_per_angstrom"],
            label=f"anchor force:{representative_id}",
            rows=int(round(num_atoms)),
        )
        stress_reference = _component_array(
            anchor["stress_components_kbar"],
            label=f"anchor stress:{representative_id}",
            rows=3,
        )
        anchor_reference = {
            "point_id": anchor["point_id"],
            "energy_ev_per_atom": _finite_float(
                anchor["energy_ev_per_atom"], "anchor energy_ev_per_atom"
            ),
            "force_components_ev_per_angstrom": force_reference,
            "stress_components_kbar": stress_reference,
            "qe_input_sha256": anchor["qe_input_sha256"],
            "qe_output_sha256": anchor["qe_output_sha256"],
        }
        point_id = _confirmation_point_id(representative_id, settings_seed)
        kpoints_grid = list(_kpoint_grid(structure, spacing))
        representative = {
            "representative_id": representative_id,
            "source_id": source_rep.get("source_id") or representative_id,
            "formula": formula,
            "elements": elements,
            "num_atoms": num_atoms,
            "source_cif": str(source_cif),
            "source_cif_sha256": source_sha,
            "copied_cif": str(copied_cif),
            "copied_cif_sha256": copied_sha,
        }
        representatives.append(representative)
        structures[representative_id] = structure
        study_points.append({
            "rank": rank,
            "point_id": point_id,
            "representative_id": representative_id,
            "source_id": representative["source_id"],
            "formula": formula,
            "num_atoms": num_atoms,
            "sweep_anchor_point_id": selection["anchor_point_id"],
            "selected_cutoff_source_point_id": selection["selected_cutoff_pair"]["point_id"],
            "selected_kpoint_source_point_id": selection["selected_kpoint"]["point_id"],
            "cutoff_pair_multiplier": cutoff_multiplier,
            "requested_kpoint_spacing_inv_angstrom": spacing,
            "kpoints_grid": kpoints_grid,
            "ecutwfc_ry": ecutwfc,
            "ecutrho_ry": ecutrho,
            "source_cif_sha256": source_sha,
            "copied_cif_sha256": copied_sha,
            "sweep_anchor_reference": anchor_reference,
        })

    covered_elements = sorted({element for rep in representatives for element in rep["elements"]})
    coverage_complete = set(covered_elements) == set(required_elements)
    if not coverage_complete:
        raise ValueError("Confirmation representatives do not cover required elements")
    selected_global_settings = {
        "cutoff_pair_multiplier": cutoff_multiplier,
        "ecutwfc_ry": ecutwfc,
        "ecutrho_ry": ecutrho,
        "kpoint_spacing_inv_angstrom": spacing,
    }
    settings_payload = {
        "workflow_version": WORKFLOW_VERSION,
        "stage": "confirmation",
        "sweep_preflight_sha256": sweep_preflight_sha,
        "source_sweep_settings_hash": sweep_settings_hash,
        "provisional_summary_sha256": _sha256(summary_snapshot),
        "provisional_results_sha256": _sha256(results_snapshot),
        "config_sha256": _sha256(config_snapshot),
        "pseudo_manifest_sha256": _sha256(pseudo_manifest_snapshot),
        "protocol_sha256": _sha256(protocol_snapshot),
        "protocol_values": sweep["protocol_values"],
        "required_elements": required_elements,
        "covered_elements": covered_elements,
        "coverage_complete": coverage_complete,
        "selected_global_settings": selected_global_settings,
        "baseline_qe_program_version": summary["qe_program_version"],
        "baseline_pw_executable_sha256": summary["pw_executable_sha256"],
        "baseline_execution_provenance": baseline_execution,
        "representatives": representatives,
        "pseudopotentials": [
            {key: entry[key] for key in ("element", "filename", "sha256")}
            for entry in bundled_pseudopotentials
        ],
        "study_points": study_points,
    }
    settings_hash = convergence_settings_hash(settings_payload)
    requested_pw = str(
        pw_executable
        or sweep.get("pw_executable_resolved")
        or sweep.get("pw_executable_requested")
        or "pw.x"
    )
    executable_path = _resolve_executable(requested_pw)
    engine_blockers = [] if executable_path else [f"pw_executable_not_found:{requested_pw}"]
    job_status = "runnable_not_started" if executable_path else "inputs_ready_engine_missing"

    queue_rows: list[dict[str, Any]] = []
    for point in study_points:
        point_id = point["point_id"]
        representative_id = point["representative_id"]
        representative = next(
            value for value in representatives
            if value["representative_id"] == representative_id
        )
        job_dir = jobs_dir / f"{int(point['rank']):03d}_{point_id}"
        job_dir.mkdir(parents=True, exist_ok=True)
        qe_scratch_outdir = _qe_scratch_outdir(
            scratch_root=resolved_scratch_root,
            campaign_namespace=scratch_campaign_namespace,
            job_id=f"{int(point['rank']):03d}_{point_id}",
        )
        input_path = job_dir / "confirmation-scf.in"
        output_path = job_dir / "confirmation-scf.out"
        run_record = job_dir / "confirmation_run.json"
        pseudo_map = {
            symbol: pseudo_by_element[symbol]["filename"]
            for symbol in representative["elements"]
        }
        pw_input = PWInput(
            structures[representative_id],
            pseudo=pseudo_map,
            control={
                "calculation": "scf",
                "restart_mode": "from_scratch",
                "prefix": point_id,
                "pseudo_dir": "../../pseudos",
                "outdir": qe_scratch_outdir,
                "disk_io": "low",
                "tstress": True,
                "tprnfor": True,
            },
            system={
                "input_dft": config["input_dft"],
                "ecutwfc": ecutwfc,
                "ecutrho": ecutrho,
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
        input_text = input_path.read_text(encoding="utf-8")
        input_text = re.sub(r"(?im)^\s*&(IONS|CELL)\s*\n\s*/\s*\n", "", input_text)
        input_path.write_text(input_text, encoding="utf-8")
        entry_id = f"confirmation:{point_id}"
        job_record = {
            "workflow_version": WORKFLOW_VERSION,
            **point,
            "kpoints_grid": point["kpoints_grid"],
            "entry_id": entry_id,
            "entry_role": "convergence_confirmation",
            "candidate_id": point_id,
            "job_status": job_status,
            "blockers": engine_blockers,
            "source_cif": representative["source_cif"],
            "copied_cif": representative["copied_cif"],
            "qe_input": str(input_path),
            "qe_input_sha256": _sha256(input_path),
            "qe_output": str(output_path),
            "run_record": str(run_record),
            "qe_scratch_outdir": qe_scratch_outdir,
            "convergence_settings_hash": settings_hash,
            "calculation_started": False,
        }
        job_record_path = job_dir / "job_plan.json"
        job_record_path.write_text(json.dumps(job_record, indent=2), encoding="utf-8")
        queue_rows.append({
            **job_record,
            "kpoints_grid": "x".join(str(value) for value in point["kpoints_grid"]),
            "sweep_anchor_reference": json.dumps(point["sweep_anchor_reference"]),
            "blockers": json.dumps(engine_blockers),
            "job_record": str(job_record_path),
        })

    queue_path = output_dir / QUEUE_NAME
    _write_csv(queue_path, queue_rows)
    preflight = {
        "workflow_version": WORKFLOW_VERSION,
        "status": job_status,
        "calculation": "scf",
        "stage": "confirmation",
        "calculation_started": False,
        "sweep_preflight": str(sweep_preflight_path),
        "sweep_preflight_sha256": sweep_preflight_sha,
        "source_sweep_settings_hash": sweep_settings_hash,
        "candidate_source_preflight": sweep.get("source_preflight"),
        "candidate_source_preflight_sha256": sweep.get("source_preflight_sha256"),
        "provisional_summary": str(summary_snapshot),
        "provisional_summary_sha256": _sha256(summary_snapshot),
        "provisional_summary_source": str(provisional_summary_path),
        "provisional_results": str(results_snapshot),
        "provisional_results_sha256": _sha256(results_snapshot),
        "provisional_results_source": str(results_path),
        "config": str(config_snapshot),
        "config_sha256": _sha256(config_snapshot),
        "pseudo_manifest": str(pseudo_manifest_snapshot),
        "pseudo_manifest_sha256": _sha256(pseudo_manifest_snapshot),
        "protocol": str(protocol_snapshot),
        "protocol_sha256": _sha256(protocol_snapshot),
        "protocol_values": sweep["protocol_values"],
        "required_elements": required_elements,
        "covered_elements": covered_elements,
        "coverage_complete": coverage_complete,
        "selected_global_settings": selected_global_settings,
        "baseline_qe_program_version": summary["qe_program_version"],
        "baseline_pw_executable_sha256": summary["pw_executable_sha256"],
        "baseline_execution_provenance": baseline_execution,
        "representatives": representatives,
        "study_points": study_points,
        "convergence_settings_payload": settings_payload,
        "convergence_settings_hash": settings_hash,
        "bundled_pseudopotentials": bundled_pseudopotentials,
        "pw_executable_requested": requested_pw,
        "pw_executable_resolved": executable_path,
        "engine_blockers": engine_blockers,
        "scratch_root": (
            str(resolved_scratch_root) if resolved_scratch_root else None
        ),
        "scratch_layout": (
            f"{scratch_campaign_namespace}/<job-id>"
            if resolved_scratch_root else "job_local_tmp"
        ),
        "scratch_campaign_namespace": (
            scratch_campaign_namespace if resolved_scratch_root else None
        ),
        "scratch_campaign_dir": (
            str(scratch_campaign_dir) if scratch_campaign_dir else None
        ),
        "resource_policy": sweep.get("resource_policy") or {},
        "queue_manifest": str(queue_path),
        "queue_manifest_sha256": _sha256(queue_path),
        "confirmation_point_count": len(queue_rows),
        "certificate_eligible": False,
        "scientific_limit": (
            "Prepared selected-combination confirmation inputs only. These jobs "
            "must complete and pass the confirmation collector before a "
            "content-hash-bound convergence certificate may authorize production "
            "DFT settings. The certificate is an integrity receipt, not a digital "
            "signature or operator-authentication mechanism."
        ),
    }
    preflight_path = output_dir / PREFLIGHT_NAME
    preflight_path.write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    print(f"Confirmation preparation status: {preflight['status']}")
    print(f"Representatives/SCF points: {len(queue_rows)}")
    print(f"Queue:     {queue_path}")
    print(f"Preflight: {preflight_path}")
    print("No Quantum ESPRESSO calculation was started.")
    return preflight


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provisional-summary", type=Path, required=True,
        help="convergence_summary.json produced by collect_qe_convergence.py",
    )
    parser.add_argument(
        "--sweep-preflight", type=Path, required=True,
        help="Original convergence_preflight.json consumed by the collector",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pw-executable", default=None,
        help="pw.x path/name; defaults to the executable requested by the sweep",
    )
    parser.add_argument(
        "--scratch-root", type=Path,
        help=(
            "Optional shared QE scratch root. Each confirmation uses a distinct "
            "<workflow>/<campaign>/<job-id> directory; default is local ./tmp."
        ),
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    prepare_confirmation_jobs(
        provisional_summary_path=args.provisional_summary,
        sweep_preflight_path=args.sweep_preflight,
        output_dir=args.output_dir,
        pw_executable=args.pw_executable,
        scratch_root=args.scratch_root,
    )
