"""Collect converged Quantum ESPRESSO static-SCF candidate energies.

This collector performs process/output/provenance gates only.  It deliberately
does not convert a static total energy into a formation energy or an
energy-above-hull value; those require consistently recomputed elemental and
competing-phase references.
"""

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

from pymatgen.core import Composition, Structure

from qe_output import summarize_qe_output


COLLECTOR_VERSION = "qe_static_collector_v1"
RY_TO_EV = 13.605693122994


def _load_json(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_static_input_hash(preflight: dict[str, Any]) -> str:
    payload = {
        "workflow_version": preflight["workflow_version"],
        "parent_relax_settings_hash": preflight["parent_relax_settings_hash"],
        "config_sha256": _sha256(Path(preflight["config"]).resolve()),
        "pseudo_manifest_sha256": _sha256(
            Path(preflight["pseudo_manifest"]).resolve()
        ),
        "global_ecutwfc_ry": preflight["global_ecutwfc_ry"],
        "global_ecutrho_ry": preflight["global_ecutrho_ry"],
    }
    computed = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    recorded = str(preflight.get("static_settings_hash") or "")
    if computed != recorded:
        raise ValueError(
            "Static preflight settings hash no longer matches its locked artifacts"
        )
    return recorded


def _bundled_pseudo_blockers(preflight: dict[str, Any]) -> list[str]:
    entries = preflight.get("bundled_pseudopotentials")
    if not isinstance(entries, list) or not entries:
        return ["bundled_pseudopotential_hash_inventory_missing"]
    blockers: list[str] = []
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
        queue_row.get("run_record") or qe_input.parent / "static_run.json"
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
    return blockers, record


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    preferred = [
        "entry_id", "entry_role", "source_id", "candidate_id", "rank",
        "formula", "composition_json", "num_atoms", "total_energy_ry",
        "total_energy_ev", "energy_ev_per_atom", "static_gate_status",
        "gate_failures", "static_input_settings_hash", "qe_program_version",
        "pw_executable_sha256",
        "static_settings_hash", "qe_input", "qe_output",
        "source_cif_sha256", "relaxed_cif_sha256", "qe_output_sha256",
        "entry_lineage_sha256", "error",
    ]
    extras = sorted({key for row in rows for key in row} - set(preferred))
    fields = preferred + extras
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def collect_static(
    *,
    preflight_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    preflight_path = Path(preflight_path).resolve()
    preflight_sha256 = _sha256(preflight_path)
    preflight = _load_json(preflight_path)
    if preflight.get("status") != "runnable_not_started":
        raise ValueError(f"Static preflight is not runnable: {preflight.get('status')}")
    if preflight.get("calculation") != "scf":
        raise ValueError("Expected a static preflight with calculation=scf")
    input_settings_hash = _verify_static_input_hash(preflight)
    queue_path = Path(preflight["queue_manifest"]).resolve()
    expected_queue_sha = str(preflight.get("queue_manifest_sha256") or "")
    if len(expected_queue_sha) != 64 or _sha256(queue_path) != expected_queue_sha:
        raise ValueError("Queue manifest no longer matches the static preflight")
    queue = _read_csv(queue_path)
    common_provenance_failures = _bundled_pseudo_blockers(preflight)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for queue_row in sorted(queue, key=lambda row: int(row["rank"])):
        candidate_id = queue_row["candidate_id"]
        entry_role = queue_row.get("entry_role") or "candidate"
        if entry_role not in {"candidate", "reference"}:
            raise ValueError(f"Unsupported entry_role for {candidate_id}: {entry_role}")
        entry_id = queue_row.get("entry_id") or f"{entry_role}:{candidate_id}"
        failures: list[str] = list(common_provenance_failures)
        if queue_row.get("static_settings_hash") != input_settings_hash:
            failures.append("queue_static_settings_hash_mismatch")
        qe_input = Path(queue_row["qe_input"]).resolve()
        qe_output = Path(queue_row["qe_output"]).resolve()
        final_cif = Path(queue_row["final_relaxed_cif"]).resolve()
        parsed = summarize_qe_output(qe_output)
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
        if not final_cif.is_file():
            failures.append("final_relaxed_cif_missing")
        else:
            expected_relaxed_sha = str(
                queue_row.get("final_relaxed_cif_sha256") or ""
            ).lower()
            if len(expected_relaxed_sha) != 64:
                failures.append("final_relaxed_cif_hash_missing")
            elif _sha256(final_cif) != expected_relaxed_sha:
                failures.append("final_relaxed_cif_hash_mismatch")
        if not parsed["output_exists"]:
            failures.append("qe_output_missing")
        if not parsed["job_done"]:
            failures.append("job_done_marker_missing")
        if parsed["electronic_convergence_failed"]:
            failures.append("electronic_convergence_failed")
        if parsed["fatal_error_detected"]:
            failures.append("fatal_qe_error_detected")
        program_version = str(parsed.get("program_version") or "").strip()
        if not program_version:
            failures.append("qe_program_version_missing")
        pw_executable_sha256 = str(
            (completed_run_record or {}).get("pw_executable_sha256") or ""
        ).lower()
        effective_settings_hash = hashlib.sha256(
            json.dumps({
                "static_input_settings_hash": input_settings_hash,
                "qe_program_version": program_version,
                "pw_executable_sha256": pw_executable_sha256,
            }, sort_keys=True).encode("utf-8")
        ).hexdigest()

        error = ""
        formula = Composition(queue_row["formula"]).reduced_formula
        composition: dict[str, float] = {}
        num_atoms: float | None = None
        try:
            if final_cif.is_file():
                structure = Structure.from_file(final_cif)
                actual_formula = structure.composition.reduced_formula
                if actual_formula != formula:
                    failures.append(
                        f"formula_mismatch:expected={formula}:actual={actual_formula}"
                    )
                composition = {
                    element.symbol: float(amount)
                    for element, amount in structure.composition.items()
                }
                num_atoms = float(structure.composition.num_atoms)
                if not math.isfinite(num_atoms) or num_atoms <= 0:
                    failures.append("invalid_atom_count")
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            failures.append("final_relaxed_cif_parse_failed")

        energy_ry = parsed.get("total_energy_ry")
        if energy_ry is None or not math.isfinite(float(energy_ry)):
            failures.append("finite_total_energy_missing")
            energy_ev = None
            energy_per_atom = None
        else:
            energy_ev = float(energy_ry) * RY_TO_EV
            energy_per_atom = (
                energy_ev / num_atoms
                if num_atoms is not None and num_atoms > 0 else None
            )
        gate_status = (
            "dft_static_converged" if not failures else "dft_static_failed_gate"
        )
        source_cif_sha256 = str(
            queue_row.get("initial_source_cif_sha256") or ""
        ).lower()
        relaxed_cif_sha256 = (
            _sha256(final_cif) if final_cif.is_file() else ""
        )
        qe_input_sha256 = _sha256(qe_input) if qe_input.is_file() else ""
        qe_output_sha256 = _sha256(qe_output) if qe_output.is_file() else ""
        entry_lineage_sha256 = hashlib.sha256(
            json.dumps({
                "entry_id": entry_id,
                "entry_role": entry_role,
                "source_id": queue_row.get("source_id") or candidate_id,
                "formula": formula,
                "composition_json": json.dumps(composition, sort_keys=True),
                "num_atoms": num_atoms,
                "source_cif_sha256": source_cif_sha256,
                "relaxed_cif_sha256": relaxed_cif_sha256,
                "qe_input_sha256": qe_input_sha256,
                "qe_output_sha256": qe_output_sha256,
                "static_settings_hash": effective_settings_hash,
                "pw_executable_sha256": pw_executable_sha256,
                "total_energy_ry": energy_ry,
            }, sort_keys=True).encode("utf-8")
        ).hexdigest()
        rows.append({
            "entry_id": entry_id,
            "entry_role": entry_role,
            "source_id": queue_row.get("source_id") or candidate_id,
            "candidate_id": candidate_id if entry_role == "candidate" else "",
            "rank": int(queue_row["rank"]),
            "formula": formula,
            "composition_json": json.dumps(composition, sort_keys=True),
            "num_atoms": num_atoms,
            "total_energy_ry": energy_ry,
            "total_energy_ev": energy_ev,
            "energy_ev_per_atom": energy_per_atom,
            "static_gate_status": gate_status,
            "gate_failures": json.dumps(failures),
            "job_done": parsed["job_done"],
            "electronic_convergence_failed": parsed["electronic_convergence_failed"],
            "fatal_error_detected": parsed["fatal_error_detected"],
            "scf_cycle_count": parsed.get("scf_cycle_count"),
            "last_scf_iteration_count": parsed.get("last_scf_iteration_count"),
            "static_input_settings_hash": input_settings_hash,
            "qe_program_version": program_version,
            "pw_executable_sha256": pw_executable_sha256,
            "static_settings_hash": effective_settings_hash,
            "source_cif_sha256": source_cif_sha256,
            "relaxed_cif_sha256": relaxed_cif_sha256,
            "qe_input": str(qe_input),
            "qe_input_sha256": qe_input_sha256,
            "qe_output": str(qe_output),
            "qe_output_sha256": qe_output_sha256,
            "entry_lineage_sha256": entry_lineage_sha256,
            "error": error,
        })

    roles = {row["entry_role"] for row in rows}
    stem = (
        "candidate_static" if roles == {"candidate"}
        else "reference_static" if roles == {"reference"}
        else "mixed_static"
    )
    manifest_path = output_dir / f"{stem}_energies.csv"
    _write_csv(manifest_path, rows)
    results_manifest_sha256 = _sha256(manifest_path)
    energy_records_sha256 = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    status_counts = Counter(row["static_gate_status"] for row in rows)
    converged_hashes = sorted({
        row["static_settings_hash"]
        for row in rows
        if row["static_gate_status"] == "dft_static_converged"
    })
    summary = {
        "collector_version": COLLECTOR_VERSION,
        "static_input_settings_hash": input_settings_hash,
        "static_settings_hash": converged_hashes[0] if len(converged_hashes) == 1 else None,
        "static_settings_hashes": converged_hashes,
        "candidate_count": len(rows),
        "status_counts": dict(status_counts),
        "dft_static_converged_count": status_counts.get("dft_static_converged", 0),
        "formation_energy_validated_count": 0,
        "convex_hull_validated_count": 0,
        "results_manifest": str(manifest_path),
        "results_manifest_sha256": results_manifest_sha256,
        "energy_records_sha256": energy_records_sha256,
        "source_inventory": preflight.get("source_inventory"),
        "source_inventory_sha256": preflight.get("source_inventory_sha256"),
        "scientific_limit": (
            "These are same-workflow candidate total energies only. Formation "
            "energy and reference-hull screening remain blocked until a complete, "
            "audited set of consistently recomputed reference phases exists."
        ),
        "candidates": rows,
    }
    summary_path = output_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Static gate counts: {dict(status_counts)}")
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
    collect_static(preflight_path=args.preflight, output_dir=args.output_dir)
