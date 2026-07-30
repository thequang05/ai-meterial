"""Collect independent QE convergence-confirmation SCFs and issue a certificate.

The selected cutoff/k-point combination is compared component-wise against the
high-cutoff/dense-grid sweep anchor for every representative.  A certificate is
written only when every output, run record, locked artifact, QE version,
executable hash, force component, stress component, and energy difference
passes the original protocol.  Partial success never produces a certificate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from collect_qe_convergence import (
    HASH_RE,
    RY_TO_EV,
    _bundled_pseudo_blockers,
    _max_component_delta,
    _parse_force_components,
    _parse_stress_kbar,
    _verify_preflight as _verify_sweep_preflight,
)
from prepare_qe_convergence_confirmation import WORKFLOW_VERSION
from prepare_qe_convergence_jobs import convergence_settings_hash, validate_protocol
from qe_convergence_certificate import (
    issue_convergence_certificate,
    verify_convergence_certificate,
)
from qe_convergence_evidence import (
    PRODUCTION_TRANSFER_SCOPE,
    derive_strictest_tested_settings,
)
from qe_output_directory import require_fresh_output_dir
from qe_execution_provenance import (
    require_same_execution_provenance,
    validate_execution_provenance,
)
from qe_output import summarize_qe_output


COLLECTOR_VERSION = "qe_convergence_confirmation_collector_v1"


def _load_json(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "rank", "point_id", "representative_id", "formula", "num_atoms",
        "ecutwfc_ry", "ecutrho_ry", "requested_kpoint_spacing_inv_angstrom",
        "kpoints_grid", "energy_ev_per_atom",
        "energy_delta_to_anchor_mev_per_atom",
        "max_force_delta_to_anchor_ev_per_angstrom",
        "max_stress_delta_to_anchor_kbar", "point_within_tolerances",
        "confirmation_gate_status", "gate_failures", "qe_program_version",
        "pw_executable_sha256", "execution_provenance",
        "qe_input_sha256", "qe_output_sha256",
    ]
    extras = sorted({key for row in rows for key in row} - set(preferred))
    fields = preferred + extras
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def _locked_file(preflight: dict[str, Any], path_key: str, hash_key: str) -> Path:
    path = Path(str(preflight.get(path_key) or "")).resolve()
    expected = str(preflight.get(hash_key) or "").lower()
    if not path.is_file() or not HASH_RE.fullmatch(expected) or _sha256(path) != expected:
        raise ValueError(f"Locked confirmation artifact hash mismatch: {path_key}")
    return path


def _verify_confirmation_preflight(
    preflight_path: Path,
) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    preflight_path = Path(preflight_path).resolve()
    preflight = _load_json(preflight_path)
    if preflight.get("workflow_version") != WORKFLOW_VERSION:
        raise ValueError("Not a qe_pbe_convergence_confirmation_v1 preflight")
    if preflight.get("status") != "runnable_not_started":
        raise ValueError(
            f"Confirmation preflight is not runnable: {preflight.get('status')}"
        )
    if preflight.get("calculation") != "scf" or preflight.get("stage") != "confirmation":
        raise ValueError("Expected a static-SCF confirmation preflight")
    if preflight.get("certificate_eligible") is not False:
        raise ValueError("Uncollected confirmation preflight has invalid certificate state")
    payload = preflight.get("convergence_settings_payload")
    settings_hash = str(preflight.get("convergence_settings_hash") or "").lower()
    if (
        not isinstance(payload, dict)
        or not HASH_RE.fullmatch(settings_hash)
        or convergence_settings_hash(payload) != settings_hash
    ):
        raise ValueError("Confirmation settings hash mismatch")
    equality_keys = (
        "required_elements", "covered_elements", "coverage_complete",
        "selected_global_settings", "representatives", "study_points",
    )
    for key in equality_keys:
        payload_key = "selected_global_settings" if key == "selected_global_settings" else key
        if payload.get(payload_key) != preflight.get(key):
            raise ValueError(f"Confirmation preflight differs from locked payload: {key}")
    if preflight.get("coverage_complete") is not True:
        raise ValueError("Confirmation element coverage is incomplete")

    sweep_path = _locked_file(preflight, "sweep_preflight", "sweep_preflight_sha256")
    sweep, sweep_sha, sweep_settings_hash = _verify_sweep_preflight(sweep_path)
    if sweep_sha != preflight["sweep_preflight_sha256"]:
        raise ValueError("Sweep preflight hash changed during confirmation verification")
    if sweep_settings_hash != preflight.get("source_sweep_settings_hash"):
        raise ValueError("Confirmation references a different sweep settings hash")
    if payload.get("source_sweep_settings_hash") != sweep_settings_hash:
        raise ValueError("Confirmation payload references a different sweep")

    artifact_specs = (
        ("provisional_summary", "provisional_summary_sha256", "provisional_summary_sha256"),
        ("provisional_results", "provisional_results_sha256", "provisional_results_sha256"),
        ("config", "config_sha256", "config_sha256"),
        ("pseudo_manifest", "pseudo_manifest_sha256", "pseudo_manifest_sha256"),
        ("protocol", "protocol_sha256", "protocol_sha256"),
    )
    for path_key, hash_key, payload_key in artifact_specs:
        _locked_file(preflight, path_key, hash_key)
        if payload.get(payload_key) != preflight.get(hash_key):
            raise ValueError(f"Confirmation payload artifact hash mismatch: {path_key}")
    protocol = validate_protocol(_load_json(Path(preflight["protocol"])))
    if protocol != preflight.get("protocol_values") or protocol != payload.get("protocol_values"):
        raise ValueError("Confirmation protocol snapshot/value mismatch")
    pseudo_failures = _bundled_pseudo_blockers(preflight)
    if pseudo_failures:
        raise ValueError("Confirmation pseudopotential verification failed: " + "; ".join(pseudo_failures))
    expected_pseudos = [
        {key: item.get(key) for key in ("element", "filename", "sha256")}
        for item in preflight["bundled_pseudopotentials"]
    ]
    if expected_pseudos != payload.get("pseudopotentials"):
        raise ValueError("Confirmation pseudopotential inventory differs from payload")
    queue_path = _locked_file(preflight, "queue_manifest", "queue_manifest_sha256")
    queue = _read_csv(queue_path)
    points = preflight.get("study_points") or []
    point_ids = [str(point.get("point_id") or "") for point in points]
    queue_ids = [row.get("point_id", "") for row in queue]
    if (
        not points
        or len(point_ids) != len(set(point_ids))
        or set(point_ids) != set(queue_ids)
        or len(queue_ids) != len(set(queue_ids))
        or preflight.get("confirmation_point_count") != len(points)
    ):
        raise ValueError("Confirmation point/queue inventory mismatch")
    return preflight, _sha256(preflight_path), settings_hash, sweep


def _queue_blockers(row: dict[str, str], point: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    exact = (
        "point_id", "representative_id", "source_id", "formula",
        "sweep_anchor_point_id", "selected_cutoff_source_point_id",
        "selected_kpoint_source_point_id", "source_cif_sha256",
        "copied_cif_sha256",
    )
    for key in exact:
        if str(row.get(key) or "") != str(point.get(key) or ""):
            failures.append(f"queue_point_mismatch:{key}")
    for key in (
        "rank", "num_atoms", "cutoff_pair_multiplier",
        "requested_kpoint_spacing_inv_angstrom", "ecutwfc_ry", "ecutrho_ry",
    ):
        try:
            left = float(row.get(key, ""))
            right = float(point[key])
            if not math.isfinite(left) or not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-10):
                failures.append(f"queue_point_mismatch:{key}")
        except (TypeError, ValueError):
            failures.append(f"queue_point_invalid:{key}")
    if row.get("kpoints_grid") != "x".join(str(value) for value in point["kpoints_grid"]):
        failures.append("queue_point_mismatch:kpoints_grid")
    expected_identity = {
        "entry_id": f"confirmation:{point['point_id']}",
        "entry_role": "convergence_confirmation",
        "candidate_id": point["point_id"],
    }
    for key, expected in expected_identity.items():
        if row.get(key) != expected:
            failures.append(f"queue_identity_mismatch:{key}")
    try:
        if json.loads(row.get("blockers") or "[]") != []:
            failures.append("queue_has_blockers")
    except json.JSONDecodeError:
        failures.append("queue_blockers_invalid")
    return failures


def _run_record_blockers(
    *, row: dict[str, str], qe_input: Path, qe_output: Path,
    parsed: dict[str, Any], preflight_sha: str, queue_sha: str,
    settings_hash: str,
) -> tuple[list[str], dict[str, Any] | None]:
    path = Path(str(row.get("run_record") or "")).resolve()
    if not path.is_file():
        return ["completed_run_record_missing"], None
    try:
        record = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return ["completed_run_record_invalid"], None
    failures: list[str] = []
    if record.get("run_status") != "completed_requires_collection":
        failures.append("completed_run_record_status_invalid")
    for key in ("entry_id", "entry_role", "source_id", "candidate_id", "formula"):
        if record.get(key) != row.get(key):
            failures.append(f"run_record_identity_mismatch:{key}")
    expected = {
        "qe_input_sha256": _sha256(qe_input) if qe_input.is_file() else "",
        "qe_output_sha256": _sha256(qe_output) if qe_output.is_file() else "",
        "preflight_sha256": preflight_sha,
        "queue_manifest_sha256": queue_sha,
        "preflight_settings_hash": settings_hash,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            failures.append(f"run_record_mismatch:{key}")
    if record.get("qe_program_version") != parsed.get("program_version"):
        failures.append("run_record_qe_version_mismatch")
    if not HASH_RE.fullmatch(str(record.get("pw_executable_sha256") or "").lower()):
        failures.append("run_record_executable_hash_missing")
    try:
        validate_execution_provenance(record.get("execution_provenance"))
    except ValueError:
        failures.append("run_record_execution_provenance_invalid")
    return failures, record


def collect_confirmation(
    *, preflight_path: Path, output_dir: Path,
) -> dict[str, Any]:
    output_dir = require_fresh_output_dir(output_dir)
    preflight_path = Path(preflight_path).resolve()
    preflight, preflight_sha, settings_hash, sweep = _verify_confirmation_preflight(
        preflight_path
    )
    queue_path = Path(preflight["queue_manifest"]).resolve()
    queue_sha = str(preflight["queue_manifest_sha256"])
    queue = _read_csv(queue_path)
    point_by_id = {point["point_id"]: point for point in preflight["study_points"]}
    protocol = preflight["protocol_values"]
    energy_tol = float(protocol["energy_tolerance_mev_per_atom"])
    force_tol = float(protocol["force_component_tolerance_ev_per_angstrom"])
    stress_tol = float(protocol["stress_component_tolerance_kbar"])
    baseline_version = str(preflight["baseline_qe_program_version"])
    baseline_executable = str(preflight["baseline_pw_executable_sha256"]).lower()
    baseline_execution = validate_execution_provenance(
        preflight.get("baseline_execution_provenance")
    )
    rows: list[dict[str, Any]] = []
    for row in sorted(queue, key=lambda item: int(item["rank"])):
        point = point_by_id[row["point_id"]]
        failures = _queue_blockers(row, point)
        if row.get("convergence_settings_hash") != settings_hash:
            failures.append("queue_convergence_settings_hash_mismatch")
        qe_input = Path(str(row.get("qe_input") or "")).resolve()
        qe_output = Path(str(row.get("qe_output") or "")).resolve()
        copied_cif = Path(str(row.get("copied_cif") or "")).resolve()
        for path, expected, label in (
            (qe_input, row.get("qe_input_sha256", ""), "qe_input"),
            (copied_cif, row.get("copied_cif_sha256", ""), "copied_cif"),
        ):
            expected = str(expected).lower()
            if not path.is_file():
                failures.append(f"{label}_missing")
            elif not HASH_RE.fullmatch(expected) or _sha256(path) != expected:
                failures.append(f"{label}_hash_mismatch")
        parsed = summarize_qe_output(qe_output)
        run_failures, record = _run_record_blockers(
            row=row, qe_input=qe_input, qe_output=qe_output, parsed=parsed,
            preflight_sha=preflight_sha, queue_sha=queue_sha,
            settings_hash=settings_hash,
        )
        failures.extend(run_failures)
        if not parsed["output_exists"]:
            failures.append("qe_output_missing")
        if not parsed["job_done"]:
            failures.append("job_done_marker_missing")
        if parsed["electronic_convergence_failed"]:
            failures.append("electronic_convergence_failed")
        if parsed["fatal_error_detected"]:
            failures.append("fatal_qe_error_detected")
        if parsed.get("last_scf_iteration_count") is None:
            failures.append("electronic_convergence_marker_missing")
        version = str(parsed.get("program_version") or "").strip()
        executable_sha = str((record or {}).get("pw_executable_sha256") or "").lower()
        if version != baseline_version:
            failures.append("qe_version_differs_from_sweep")
        if executable_sha != baseline_executable:
            failures.append("qe_executable_differs_from_sweep")
        try:
            execution = require_same_execution_provenance(
                (record or {}).get("execution_provenance"),
                baseline_execution,
                label="convergence sweep",
            )
        except ValueError:
            failures.append("execution_provenance_differs_from_sweep")
            execution = None
        num_atoms_float = float(point["num_atoms"])
        num_atoms = int(round(num_atoms_float))
        if num_atoms < 1 or not math.isclose(num_atoms_float, num_atoms, abs_tol=1e-9):
            failures.append("canonical_atom_count_invalid")
            num_atoms = max(num_atoms, 1)
        total_energy = parsed.get("total_energy_ry")
        if total_energy is None or not math.isfinite(float(total_energy)):
            failures.append("finite_total_energy_missing")
            energy_per_atom: float | None = None
        else:
            energy_per_atom = float(total_energy) * RY_TO_EV / num_atoms
        try:
            forces = _parse_force_components(qe_output, num_atoms)
        except (OSError, ValueError) as exc:
            failures.append(f"force_components_invalid:{exc}")
            forces = None
        try:
            stress = _parse_stress_kbar(qe_output)
        except (OSError, ValueError) as exc:
            failures.append(f"stress_components_invalid:{exc}")
            stress = None
        anchor = point["sweep_anchor_reference"]
        energy_delta: float | None = None
        force_delta: float | None = None
        stress_delta: float | None = None
        if energy_per_atom is not None:
            energy_delta = abs(energy_per_atom - float(anchor["energy_ev_per_atom"])) * 1000.0
        if forces is not None:
            force_delta = _max_component_delta(
                forces, anchor["force_components_ev_per_angstrom"]
            )
        if stress is not None:
            stress_delta = _max_component_delta(stress, anchor["stress_components_kbar"])
        within = (
            not failures
            and energy_delta is not None and energy_delta <= energy_tol
            and force_delta is not None and force_delta <= force_tol
            and stress_delta is not None and stress_delta <= stress_tol
        )
        if energy_delta is not None and energy_delta > energy_tol:
            failures.append("energy_delta_above_tolerance")
        if force_delta is not None and force_delta > force_tol:
            failures.append("force_component_delta_above_tolerance")
        if stress_delta is not None and stress_delta > stress_tol:
            failures.append("stress_component_delta_above_tolerance")
        gate_status = "confirmation_passed" if within else "confirmation_failed_gate"
        rows.append({
            **{key: value for key, value in point.items() if key != "sweep_anchor_reference"},
            "kpoints_grid": "x".join(str(value) for value in point["kpoints_grid"]),
            "energy_ev_per_atom": energy_per_atom if energy_per_atom is not None else "",
            "energy_delta_to_anchor_mev_per_atom": energy_delta if energy_delta is not None else "",
            "max_force_delta_to_anchor_ev_per_angstrom": force_delta if force_delta is not None else "",
            "max_stress_delta_to_anchor_kbar": stress_delta if stress_delta is not None else "",
            "force_components_ev_per_angstrom": json.dumps(forces) if forces is not None else "",
            "stress_components_kbar": json.dumps(stress) if stress is not None else "",
            "point_within_tolerances": within,
            "confirmation_gate_status": gate_status,
            "gate_failures": json.dumps(sorted(set(failures))),
            "qe_program_version": version,
            "pw_executable_sha256": executable_sha,
            "execution_provenance": (
                json.dumps(execution, sort_keys=True) if execution else ""
            ),
            "qe_input_sha256": _sha256(qe_input) if qe_input.is_file() else "",
            "qe_output_sha256": _sha256(qe_output) if qe_output.is_file() else "",
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "confirmation_results.csv"
    _write_csv(results_path, rows)
    counts = Counter(row["confirmation_gate_status"] for row in rows)
    all_passed = bool(rows) and counts.get("confirmation_passed", 0) == len(rows)
    status = "confirmation_passed" if all_passed else "blocked_confirmation_failed"
    summary = {
        "collector_version": COLLECTOR_VERSION,
        "status": status,
        "confirmation_settings_hash": settings_hash,
        "source_sweep_settings_hash": preflight["source_sweep_settings_hash"],
        "point_count": len(rows),
        "gate_status_counts": dict(counts),
        "coverage_complete": preflight.get("coverage_complete") is True,
        "qe_program_version": baseline_version if all_passed else None,
        "pw_executable_sha256": baseline_executable if all_passed else None,
        "execution_provenance": baseline_execution if all_passed else None,
        "selected_settings": preflight["selected_global_settings"],
        "certificate_eligible": all_passed,
        "confirmation_required": not all_passed,
        "confirmation_preflight": str(preflight_path),
        "confirmation_preflight_sha256": preflight_sha,
        "queue_manifest": str(queue_path),
        "queue_manifest_sha256": queue_sha,
        "results": str(results_path),
        "results_sha256": _sha256(results_path),
        "scientific_limit": (
            "Confirmation establishes numerical convergence only inside the tested "
            "finite-smearing PBE window; it is not a material-stability result."
        ),
        "points": rows,
    }
    summary_path = output_dir / "confirmation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    certificate_path = output_dir / "qe_convergence_certificate.json"
    if all_passed:
        production_settings = derive_strictest_tested_settings(
            protocol=preflight["protocol_values"],
            base_ecutwfc_ry=sweep["base_ecutwfc_ry"],
            base_ecutrho_ry=sweep["base_ecutrho_ry"],
        )
        representatives = [
            {
                "representative_id": row["representative_id"],
                "point_id": row["point_id"],
                "status": "passed",
                "energy_delta_to_anchor_mev_per_atom": row[
                    "energy_delta_to_anchor_mev_per_atom"
                ],
                "max_force_delta_to_anchor_ev_per_angstrom": row[
                    "max_force_delta_to_anchor_ev_per_angstrom"
                ],
                "max_stress_delta_to_anchor_kbar": row[
                    "max_stress_delta_to_anchor_kbar"
                ],
            }
            for row in rows
        ]
        artifacts = {
            "sweep_preflight": {
                "path": str(Path(preflight["sweep_preflight"]).resolve()),
                "sha256": preflight["sweep_preflight_sha256"],
            },
            "sweep_results": {
                "path": str(Path(preflight["provisional_results"]).resolve()),
                "sha256": preflight["provisional_results_sha256"],
            },
            "sweep_summary": {
                "path": str(Path(preflight["provisional_summary"]).resolve()),
                "sha256": preflight["provisional_summary_sha256"],
            },
            "confirmation_preflight": {
                "path": str(preflight_path), "sha256": preflight_sha,
            },
            "confirmation_queue": {
                "path": str(queue_path), "sha256": queue_sha,
            },
            "confirmation_results": {
                "path": str(results_path), "sha256": _sha256(results_path),
            },
            "confirmation_summary": {
                "path": str(summary_path), "sha256": _sha256(summary_path),
            },
            "config": {
                "path": str(Path(preflight["config"]).resolve()),
                "sha256": preflight["config_sha256"],
            },
            "pseudo_manifest": {
                "path": str(Path(preflight["pseudo_manifest"]).resolve()),
                "sha256": preflight["pseudo_manifest_sha256"],
            },
            "protocol": {
                "path": str(Path(preflight["protocol"]).resolve()),
                "sha256": preflight["protocol_sha256"],
            },
        }
        payload = {
            "target_profile": "finite_smearing_static_scf",
            "source_sweep_settings_hash": preflight["source_sweep_settings_hash"],
            "confirmation_settings_hash": settings_hash,
            "base_ecutwfc_ry": sweep["base_ecutwfc_ry"],
            "base_ecutrho_ry": sweep["base_ecutrho_ry"],
            "required_elements": preflight["required_elements"],
            "covered_elements": preflight["covered_elements"],
            "coverage_complete": True,
            "selected_settings": preflight["selected_global_settings"],
            "production_settings": production_settings,
            "production_settings_scope": PRODUCTION_TRANSFER_SCOPE,
            "qe_program_version": baseline_version,
            "pw_executable_sha256": baseline_executable,
            "execution_provenance": baseline_execution,
            "artifacts": artifacts,
            "pseudopotentials": [
                {
                    "element": entry["element"],
                    "filename": entry["filename"],
                    "path": entry["path"],
                    "sha256": entry["sha256"],
                }
                for entry in preflight["bundled_pseudopotentials"]
            ],
            "representatives": representatives,
        }
        pending_certificate_path = output_dir / ".qe_convergence_certificate.pending.json"
        if certificate_path.exists() or pending_certificate_path.exists():
            raise FileExistsError(
                "Refusing to overwrite an existing confirmation certificate artifact"
            )
        issue_convergence_certificate(
            payload=payload, output_path=pending_certificate_path
        )
        verify_convergence_certificate(
            pending_certificate_path,
            required_elements=preflight["required_elements"],
            config_path=Path(preflight["config"]),
            pseudo_manifest_path=Path(preflight["pseudo_manifest"]),
        )
        pending_certificate_path.replace(certificate_path)
    elif certificate_path.exists():
        raise FileExistsError(
            "A failed confirmation output directory already contains a certificate"
        )
    print(f"Confirmation status: {status}")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    if all_passed:
        print(f"Certificate: {certificate_path}")
    else:
        print("Certificate: not issued")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    collect_confirmation(preflight_path=args.preflight, output_dir=args.output_dir)
