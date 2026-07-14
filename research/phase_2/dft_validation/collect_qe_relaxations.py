"""Collect and gate completed Quantum ESPRESSO ``vc-relax`` calculations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read as ase_read
from pymatgen.core import Composition, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter

from qe_output import summarize_qe_output
from qe_convergence_certificate import verify_convergence_certificate
from qe_execution_provenance import (
    require_same_execution_provenance,
    validate_execution_provenance,
)
from qe_output_directory import require_fresh_output_dir


COLLECTOR_VERSION = "qe_relax_collector_v1"
RY_TO_EV = 13.605693122994
BOHR_TO_ANGSTROM = 0.529177210903
EV_PER_ANGSTROM3_TO_KBAR = 1602.1766208


def _load_json(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).resolve().open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _settings_hash(preflight: dict[str, Any]) -> str:
    config_path = Path(preflight["config"]).resolve()
    pseudo_path = Path(preflight["pseudo_manifest"]).resolve()
    payload = {
        "workflow_version": preflight["workflow_version"],
        "config_sha256": _sha256(config_path),
        "pseudo_manifest_sha256": _sha256(pseudo_path),
        "global_ecutwfc_ry": preflight["global_ecutwfc_ry"],
        "global_ecutrho_ry": preflight["global_ecutrho_ry"],
        "effective_kpoint_spacing_inv_angstrom": preflight[
            "effective_kpoint_spacing_inv_angstrom"
        ],
        "convergence_certificate_payload_sha256": preflight[
            "convergence_certificate_payload_sha256"
        ],
    }
    computed = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    recorded = preflight.get("relax_settings_hash")
    if recorded and recorded != computed:
        raise ValueError(
            "Relax preflight settings hash no longer matches its locked artifacts"
        )
    return str(recorded or computed)


def _bundled_pseudo_blockers(preflight: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    entries = preflight.get("bundled_pseudopotentials")
    if not isinstance(entries, list) or not entries:
        return ["bundled_pseudopotential_hash_inventory_missing"]
    for entry in entries:
        label = str(entry.get("element") or entry.get("filename") or "unknown")
        path = Path(str(entry.get("path") or "")).resolve()
        expected = str(entry.get("sha256") or "").lower()
        if not path.is_file():
            blockers.append(f"bundled_pseudopotential_missing:{label}")
        elif len(expected) != 64 or _sha256(path) != expected:
            blockers.append(f"bundled_pseudopotential_hash_mismatch:{label}")
    return blockers


def _run_record_blockers(
    *,
    queue_row: dict[str, str],
    qe_input: Path,
    qe_output: Path,
    parsed: dict[str, Any],
    preflight_sha256: str,
    queue_manifest_sha256: str,
    input_settings_hash: str,
) -> tuple[list[str], dict[str, Any] | None]:
    record_path = Path(
        queue_row.get("run_record") or qe_input.parent / "job_run.json"
    ).resolve()
    if not record_path.is_file():
        return ["completed_run_record_missing"], None
    try:
        record = _load_json(record_path)
    except (ValueError, json.JSONDecodeError, OSError):
        return ["completed_run_record_invalid"], None
    blockers: list[str] = []
    if record.get("run_status") != "completed_requires_collection":
        blockers.append("completed_run_record_status_invalid")
    for key, fallback in (
        ("entry_id", f"candidate:{queue_row['candidate_id']}"),
        ("entry_role", "candidate"),
        ("source_id", queue_row["candidate_id"]),
        ("candidate_id", queue_row["candidate_id"]),
        ("formula", queue_row["formula"]),
    ):
        if record.get(key) != (queue_row.get(key) or fallback):
            blockers.append(f"run_record_identity_mismatch:{key}")
    actual_input_sha = _sha256(qe_input) if qe_input.is_file() else ""
    actual_output_sha = _sha256(qe_output) if qe_output.is_file() else ""
    if record.get("qe_input_sha256") != actual_input_sha:
        blockers.append("run_record_input_hash_mismatch")
    if record.get("qe_output_sha256") != actual_output_sha:
        blockers.append("run_record_output_hash_mismatch")
    if record.get("preflight_sha256") != preflight_sha256:
        blockers.append("run_record_preflight_hash_mismatch")
    if record.get("queue_manifest_sha256") != queue_manifest_sha256:
        blockers.append("run_record_queue_hash_mismatch")
    if record.get("preflight_settings_hash") != input_settings_hash:
        blockers.append("run_record_settings_hash_mismatch")
    if record.get("qe_program_version") != parsed.get("program_version"):
        blockers.append("run_record_qe_version_mismatch")
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(record.get("pw_executable_sha256") or "").lower()
    ):
        blockers.append("run_record_executable_hash_missing")
    try:
        validate_execution_provenance(record.get("execution_provenance"))
    except ValueError:
        blockers.append("run_record_execution_provenance_invalid")
    return blockers, record


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    preferred = [
        "rank", "entry_id", "entry_role", "source_id", "candidate_id",
        "formula", "relax_gate_status", "gate_failures",
        "job_done", "ionic_converged_marker", "electronic_convergence_failed",
        "fatal_error_detected", "final_energy_ev", "final_energy_ev_per_atom",
        "final_max_force_component_ev_per_angstrom",
        "final_max_force_norm_ev_per_angstrom", "final_pressure_kbar",
        "ase_hydrostatic_pressure_kbar", "max_abs_stress_component_kbar",
        "max_abs_deviatoric_stress_component_kbar", "final_volume_angstrom3",
        "num_sites", "initial_source_cif_sha256", "final_cif",
        "final_cif_sha256", "qe_input", "qe_output",
        "relax_input_settings_hash", "qe_program_version",
        "pw_executable_sha256", "execution_provenance", "settings_hash",
        "error",
    ]
    extras = sorted({key for row in rows for key in row} - set(preferred))
    fields = preferred + extras
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def collect_relaxations(
    *,
    preflight_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = require_fresh_output_dir(output_dir)
    preflight_path = Path(preflight_path).resolve()
    preflight_sha256 = _sha256(preflight_path)
    preflight = _load_json(preflight_path)
    if preflight.get("status") != "runnable_not_started":
        raise ValueError(f"Preflight is not runnable: {preflight.get('status')}")
    if preflight.get("production_settings_certified") is not True:
        raise ValueError("Relax preflight is not convergence-certified")
    certificate = verify_convergence_certificate(
        Path(str(preflight.get("convergence_certificate") or "")),
        required_elements=preflight.get("required_elements") or [],
        config_path=Path(preflight["config"]),
        pseudo_manifest_path=Path(preflight["pseudo_manifest"]),
    )
    if (
        preflight.get("convergence_certificate_sha256")
        != certificate["certificate_sha256"]
        or preflight.get("convergence_certificate_payload_sha256")
        != certificate["certificate_payload_sha256"]
    ):
        raise ValueError("Relax preflight convergence-certificate lineage mismatch")
    config = _load_json(Path(preflight["config"]))
    queue_path = Path(preflight["queue_manifest"]).resolve()
    expected_queue_sha = str(preflight.get("queue_manifest_sha256") or "")
    if len(expected_queue_sha) != 64 or _sha256(queue_path) != expected_queue_sha:
        raise ValueError("Queue manifest no longer matches the relax preflight")
    queue = _read_csv(queue_path)
    input_settings_hash = _settings_hash(preflight)
    common_provenance_failures = _bundled_pseudo_blockers(preflight)
    cif_dir = output_dir / "relaxed_cifs"
    cif_dir.mkdir(parents=True, exist_ok=True)

    force_threshold = (
        float(config["forc_conv_thr_ry_per_bohr"]) * RY_TO_EV / BOHR_TO_ANGSTROM
    )
    pressure_threshold = float(config["press_conv_thr_kbar"])
    rows: list[dict[str, Any]] = []
    for queue_row in sorted(queue, key=lambda row: int(row["rank"])):
        candidate_id = queue_row["candidate_id"]
        qe_input = Path(queue_row["qe_input"]).resolve()
        qe_output = (
            Path(queue_row["qe_output"]).resolve()
            if queue_row.get("qe_output")
            else qe_input.parent / "vc-relax.out"
        )
        parsed = summarize_qe_output(qe_output)
        failures: list[str] = list(common_provenance_failures)
        run_record_blockers, completed_run_record = _run_record_blockers(
            queue_row=queue_row,
            qe_input=qe_input,
            qe_output=qe_output,
            parsed=parsed,
            preflight_sha256=preflight_sha256,
            queue_manifest_sha256=expected_queue_sha,
            input_settings_hash=input_settings_hash,
        )
        failures.extend(run_record_blockers)
        expected_input_sha = str(queue_row.get("qe_input_sha256") or "").lower()
        if not qe_input.is_file():
            failures.append("qe_input_missing")
        elif len(expected_input_sha) != 64:
            failures.append("qe_input_hash_missing")
        elif _sha256(qe_input) != expected_input_sha:
            failures.append("qe_input_hash_mismatch")
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
        if not parsed.get("ionic_converged_marker", False):
            failures.append("ionic_convergence_marker_missing")
        program_version = str(parsed.get("program_version") or "").strip()
        if not program_version:
            failures.append("qe_program_version_missing")
        pw_executable_sha256 = str(
            (completed_run_record or {}).get("pw_executable_sha256") or ""
        ).lower()
        if program_version != certificate["qe_program_version"]:
            failures.append("qe_version_differs_from_convergence_certificate")
        if pw_executable_sha256 != certificate["pw_executable_sha256"]:
            failures.append("qe_executable_differs_from_convergence_certificate")
        try:
            execution = require_same_execution_provenance(
                (completed_run_record or {}).get("execution_provenance"),
                certificate["execution_provenance"],
                label="convergence certificate",
            )
        except ValueError:
            failures.append("execution_provenance_differs_from_convergence_certificate")
            execution = None
        effective_settings_hash = hashlib.sha256(
            json.dumps({
                "relax_input_settings_hash": input_settings_hash,
                "qe_program_version": program_version,
                "pw_executable_sha256": pw_executable_sha256,
            }, sort_keys=True).encode("utf-8")
        ).hexdigest()

        error = ""
        final_structure: Structure | None = None
        final_energy_ev: float | None = None
        max_force_component: float | None = None
        max_force_norm: float | None = None
        stress_kbar: np.ndarray | None = None
        ase_pressure_kbar: float | None = None
        max_stress_component_kbar: float | None = None
        max_deviatoric_stress_kbar: float | None = None
        if parsed["output_exists"] and not parsed["fatal_error_detected"]:
            try:
                atoms = ase_read(
                    str(qe_output), index=-1, format="espresso-out", results_required=True
                )
                final_structure = AseAtomsAdaptor.get_structure(atoms)
                final_energy_ev = float(atoms.get_potential_energy())
                forces = np.asarray(atoms.get_forces(), dtype=float)
                max_force_component = float(np.max(np.abs(forces)))
                max_force_norm = float(np.max(np.linalg.norm(forces, axis=1)))
                stress_ev_per_angstrom3 = np.asarray(
                    atoms.get_stress(voigt=False), dtype=float
                )
                if stress_ev_per_angstrom3.shape != (3, 3) or not np.all(
                    np.isfinite(stress_ev_per_angstrom3)
                ):
                    raise ValueError("ASE returned an invalid final stress tensor")
                stress_kbar = stress_ev_per_angstrom3 * EV_PER_ANGSTROM3_TO_KBAR
                ase_pressure_kbar = float(-np.trace(stress_kbar) / 3.0)
                deviatoric = stress_kbar + np.eye(3) * ase_pressure_kbar
                max_stress_component_kbar = float(np.max(np.abs(stress_kbar)))
                max_deviatoric_stress_kbar = float(np.max(np.abs(deviatoric)))
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                failures.append("final_structure_or_forces_parse_failed")

        expected_formula = Composition(queue_row["formula"]).reduced_formula
        actual_formula = (
            final_structure.composition.reduced_formula if final_structure else ""
        )
        if final_structure and actual_formula != expected_formula:
            failures.append(
                f"formula_mismatch:expected={expected_formula}:actual={actual_formula}"
            )
        final_volume = float(final_structure.volume) if final_structure else None
        if final_structure and (
            len(final_structure) < 1
            or final_volume is None
            or not math.isfinite(final_volume)
            or final_volume <= 0
            or not all(
                math.isfinite(float(length)) and float(length) > 0
                for length in final_structure.lattice.abc
            )
        ):
            failures.append("invalid_final_cell")
        parsed_energy_ry = parsed.get("total_energy_ry")
        if final_energy_ev is None or not math.isfinite(final_energy_ev):
            failures.append("finite_final_energy_missing")
        if parsed_energy_ry is None or not math.isfinite(float(parsed_energy_ry)):
            failures.append("finite_qe_total_energy_missing")
        elif final_energy_ev is not None and math.isfinite(final_energy_ev):
            parsed_energy_ev = float(parsed_energy_ry) * RY_TO_EV
            tolerance_ev = max(1e-3, abs(parsed_energy_ev) * 1e-8)
            if abs(final_energy_ev - parsed_energy_ev) > tolerance_ev:
                failures.append("ase_qe_total_energy_mismatch")
        if max_force_component is not None and (
            not math.isfinite(max_force_component)
            or max_force_component > force_threshold * 1.05
        ):
            failures.append("force_component_above_threshold")
        pressure = parsed.get("pressure_kbar")
        if pressure is None:
            failures.append("final_pressure_missing")
        elif not math.isfinite(float(pressure)) or abs(float(pressure)) > pressure_threshold * 1.05:
            failures.append("pressure_above_threshold")
        if stress_kbar is None:
            failures.append("final_stress_tensor_missing")
        elif (
            max_stress_component_kbar is None
            or not math.isfinite(max_stress_component_kbar)
            or max_stress_component_kbar > pressure_threshold * 1.05
        ):
            failures.append("stress_tensor_component_above_threshold")

        final_cif = cif_dir / f"{candidate_id}_qe_relaxed.cif"
        if final_structure is not None and not failures:
            clean = final_structure.copy().relabel_sites()
            CifWriter(clean).write_file(str(final_cif))
            gate_status = "dft_relax_converged"
        else:
            gate_status = "dft_relax_failed_gate"

        num_sites = len(final_structure) if final_structure else ""
        energy_per_atom = (
            final_energy_ev / len(final_structure)
            if final_energy_ev is not None and final_structure else None
        )
        rows.append({
            "rank": int(queue_row["rank"]),
            "entry_id": queue_row.get("entry_id") or f"candidate:{candidate_id}",
            "entry_role": queue_row.get("entry_role") or "candidate",
            "source_id": queue_row.get("source_id") or candidate_id,
            "candidate_id": candidate_id,
            "formula": expected_formula,
            "relax_gate_status": gate_status,
            "gate_failures": json.dumps(failures),
            **parsed,
            "final_energy_ev": final_energy_ev,
            "final_energy_ev_per_atom": energy_per_atom,
            "final_max_force_component_ev_per_angstrom": max_force_component,
            "final_max_force_norm_ev_per_angstrom": max_force_norm,
            "final_pressure_kbar": pressure,
            "ase_hydrostatic_pressure_kbar": ase_pressure_kbar,
            "max_abs_stress_component_kbar": max_stress_component_kbar,
            "max_abs_deviatoric_stress_component_kbar": max_deviatoric_stress_kbar,
            "final_stress_tensor_kbar": (
                json.dumps(stress_kbar.tolist()) if stress_kbar is not None else ""
            ),
            "force_component_threshold_ev_per_angstrom": force_threshold,
            "pressure_threshold_kbar": pressure_threshold,
            "num_sites": num_sites,
            "final_volume_angstrom3": final_volume,
            "final_cif": str(final_cif) if gate_status == "dft_relax_converged" else "",
            "initial_source_cif_sha256": queue_row.get("source_cif_sha256") or "",
            "final_cif_sha256": (
                _sha256(final_cif)
                if gate_status == "dft_relax_converged" and final_cif.is_file()
                else ""
            ),
            "qe_input": str(qe_input),
            "qe_output": str(qe_output),
            "qe_input_sha256": _sha256(qe_input) if qe_input.is_file() else "",
            "qe_output_sha256": _sha256(qe_output) if qe_output.is_file() else "",
            "relax_input_settings_hash": input_settings_hash,
            "qe_program_version": program_version,
            "pw_executable_sha256": pw_executable_sha256,
            "execution_provenance": (
                json.dumps(execution, sort_keys=True) if execution else ""
            ),
            "settings_hash": effective_settings_hash,
            "convergence_certificate_id": certificate["certificate_id"],
            "convergence_certificate_payload_sha256": certificate[
                "certificate_payload_sha256"
            ],
            "error": error,
        })

    manifest_path = output_dir / "qe_relaxation_results.csv"
    _write_csv(manifest_path, rows)
    results_manifest_sha256 = _sha256(manifest_path)
    status_counts = Counter(row["relax_gate_status"] for row in rows)
    converged_hashes = sorted({
        row["settings_hash"]
        for row in rows
        if row["relax_gate_status"] == "dft_relax_converged"
    })
    summary = {
        "collector_version": COLLECTOR_VERSION,
        "source_preflight": str(preflight_path),
        "source_preflight_sha256": preflight_sha256,
        "source_queue_manifest": str(queue_path),
        "source_queue_manifest_sha256": expected_queue_sha,
        "relax_input_settings_hash": input_settings_hash,
        "convergence_certificate": certificate["certificate_path"],
        "convergence_certificate_sha256": certificate["certificate_sha256"],
        "convergence_certificate_id": certificate["certificate_id"],
        "convergence_certificate_payload_sha256": certificate[
            "certificate_payload_sha256"
        ],
        "execution_provenance": certificate["execution_provenance"],
        "settings_hash": converged_hashes[0] if len(converged_hashes) == 1 else None,
        "settings_hashes": converged_hashes,
        "candidate_count": len(rows),
        "status_counts": dict(status_counts),
        "dft_relax_converged_count": status_counts.get("dft_relax_converged", 0),
        "thermodynamically_validated_count": 0,
        "results_manifest": str(manifest_path),
        "results_manifest_sha256": results_manifest_sha256,
        "scientific_limit": (
            "A converged DFT relaxation validates the local geometry only. "
            "Static energies and competing-phase convex-hull calculations remain required."
        ),
        "candidates": rows,
    }
    summary_path = output_dir / "qe_relaxation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Relaxation gate counts: {dict(status_counts)}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary:  {summary_path}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    collect_relaxations(preflight_path=args.preflight, output_dir=args.output_dir)
