"""Prepare (but never execute) consistent Quantum ESPRESSO static-SCF jobs.

Only candidates that passed ``collect_qe_relaxations.py`` are accepted.  The
same PBE pseudopotentials and global cutoffs used for ``vc-relax`` are verified
again, while the tighter static k-point spacing and electronic threshold come
from the locked workflow config.  This module writes inputs and an auditable
queue; it never launches ``pw.x``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

from pymatgen.core import Composition, Structure
from pymatgen.io.pwscf import PWInput

from prepare_qe_jobs import (
    _kpoint_grid,
    _resolve_executable,
    _sha256,
    _validate_config,
    _validate_pseudopotentials,
)


WORKFLOW_VERSION = "qe_pbe_candidate_static_v1"


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    preferred = [
        "rank", "entry_id", "entry_role", "source_id", "candidate_id",
        "formula", "job_status", "blockers",
        "final_relaxed_cif", "kpoints_grid", "qe_input", "qe_output",
        "qe_input_sha256", "run_record", "parent_relax_settings_hash",
        "static_settings_hash",
    ]
    extras = sorted({key for row in rows for key in row} - set(preferred))
    fields = preferred + extras
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def _relax_settings_hash(preflight: dict[str, Any]) -> str:
    config_path = Path(preflight["config"]).resolve()
    pseudo_path = Path(preflight["pseudo_manifest"]).resolve()
    payload = {
        "workflow_version": preflight["workflow_version"],
        "config_sha256": _sha256(config_path),
        "pseudo_manifest_sha256": _sha256(pseudo_path),
        "global_ecutwfc_ry": preflight["global_ecutwfc_ry"],
        "global_ecutrho_ry": preflight["global_ecutrho_ry"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _static_settings_hash(
    *,
    parent_relax_settings_hash: str,
    config_path: Path,
    pseudo_manifest_path: Path,
    global_ecutwfc_ry: float,
    global_ecutrho_ry: float,
) -> str:
    payload = {
        "workflow_version": WORKFLOW_VERSION,
        "parent_relax_settings_hash": parent_relax_settings_hash,
        "config_sha256": _sha256(config_path),
        "pseudo_manifest_sha256": _sha256(pseudo_manifest_path),
        "global_ecutwfc_ry": global_ecutwfc_ry,
        "global_ecutrho_ry": global_ecutrho_ry,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def prepare_static_jobs(
    *,
    relaxation_results_path: Path,
    relax_preflight_path: Path,
    output_dir: Path,
    pw_executable: str | None = None,
) -> dict[str, Any]:
    relax_preflight_path = Path(relax_preflight_path).resolve()
    relax_preflight = _load_json(relax_preflight_path)
    if relax_preflight.get("status") != "runnable_not_started":
        raise ValueError(
            "Relax preflight must be runnable_not_started; got "
            f"{relax_preflight.get('status')}"
        )
    if not relax_preflight.get("pseudo_manifest"):
        raise ValueError("Relax preflight has no locked pseudopotential manifest")

    config_path = Path(relax_preflight["config"]).resolve()
    config = _load_json(config_path)
    _validate_config(config)
    for field in (
        "static_kpoint_spacing_inv_angstrom",
        "static_degauss_ry",
        "static_conv_thr",
    ):
        value = float(config.get(field, 0))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Config field must be positive: {field}")

    all_rows = _read_csv(relaxation_results_path)
    converged = [
        row for row in all_rows
        if row.get("relax_gate_status") == "dft_relax_converged"
    ]
    if not converged:
        raise ValueError("No dft_relax_converged candidates were found")

    expected_parent_input_hash = _relax_settings_hash(relax_preflight)
    observed_input_hashes = {
        row.get("relax_input_settings_hash", "") for row in converged
    }
    if observed_input_hashes != {expected_parent_input_hash}:
        raise ValueError(
            "Relaxation results do not match the supplied relax preflight: "
            f"expected input settings hash={expected_parent_input_hash}, "
            f"observed={sorted(observed_input_hashes)}"
        )
    effective_parent_hashes = {row.get("settings_hash", "") for row in converged}
    if len(effective_parent_hashes) != 1:
        raise ValueError(
            "Relaxed candidates use mixed QE versions/settings: "
            f"{sorted(effective_parent_hashes)}"
        )
    expected_parent_hash = next(iter(effective_parent_hashes))
    if len(expected_parent_hash) != 64:
        raise ValueError("Relaxation results have no valid effective settings hash")

    structures: list[tuple[dict[str, str], Structure, Path]] = []
    required_elements_set: set[str] = set()
    for row in converged:
        final_cif = Path(row["final_cif"]).resolve()
        if not final_cif.is_file():
            raise FileNotFoundError(final_cif)
        initial_source_sha = str(row.get("initial_source_cif_sha256") or "").lower()
        if len(initial_source_sha) != 64:
            raise ValueError(
                f"Relaxation result lacks source CIF lineage for {row['candidate_id']}"
            )
        expected_final_sha = str(row.get("final_cif_sha256") or "").lower()
        if len(expected_final_sha) != 64 or _sha256(final_cif) != expected_final_sha:
            raise ValueError(
                f"Relaxed CIF hash mismatch for {row['candidate_id']}"
            )
        structure = Structure.from_file(final_cif)
        expected_formula = Composition(row["formula"]).reduced_formula
        if structure.composition.reduced_formula != expected_formula:
            raise ValueError(
                f"Final CIF formula mismatch for {row['candidate_id']}: "
                f"expected={expected_formula}, "
                f"actual={structure.composition.reduced_formula}"
            )
        required_elements_set.update(
            element.symbol for element in structure.composition.elements
        )
        structures.append((row, structure, final_cif))

    required_elements = sorted(
        required_elements_set,
        key=lambda symbol: Composition(symbol).elements[0].Z,
    )
    pseudo_manifest_path = Path(relax_preflight["pseudo_manifest"]).resolve()
    relax_pseudo_dir = relax_preflight_path.parent / "pseudos"
    pseudo_entries, pseudo_blockers, _ = _validate_pseudopotentials(
        required_elements=required_elements,
        pseudo_manifest_path=pseudo_manifest_path,
        pseudo_dir=relax_pseudo_dir,
    )
    if pseudo_blockers:
        raise ValueError(
            "Bundled relaxation pseudopotentials failed re-verification: "
            + "; ".join(pseudo_blockers)
        )

    global_ecutwfc = float(relax_preflight["global_ecutwfc_ry"])
    global_ecutrho = float(relax_preflight["global_ecutrho_ry"])
    if global_ecutwfc < max(
        float(entry["ecutwfc_ry"]) for entry in pseudo_entries.values()
    ):
        raise ValueError("Relax global ecutwfc is below a verified recommendation")
    if global_ecutrho < max(
        float(entry["ecutrho_ry"]) for entry in pseudo_entries.values()
    ):
        raise ValueError("Relax global ecutrho is below a verified recommendation")

    requested_executable = (
        pw_executable
        or relax_preflight.get("pw_executable_resolved")
        or relax_preflight.get("pw_executable_requested")
        or "pw.x"
    )
    executable_path = _resolve_executable(str(requested_executable))
    engine_blockers = (
        [] if executable_path else [f"pw_executable_not_found:{requested_executable}"]
    )

    output_dir = Path(output_dir).resolve()
    jobs_dir = output_dir / "jobs"
    bundled_pseudo_dir = output_dir / "pseudos"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    bundled_pseudo_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot_path = output_dir / "workflow_config_snapshot.json"
    pseudo_manifest_snapshot_path = output_dir / "pseudo_manifest_snapshot.json"
    shutil.copy2(config_path, config_snapshot_path)
    shutil.copy2(pseudo_manifest_path, pseudo_manifest_snapshot_path)
    for entry in pseudo_entries.values():
        shutil.copy2(entry["source_path"], bundled_pseudo_dir / entry["filename"])

    static_hash = _static_settings_hash(
        parent_relax_settings_hash=expected_parent_hash,
        config_path=config_snapshot_path,
        pseudo_manifest_path=pseudo_manifest_snapshot_path,
        global_ecutwfc_ry=global_ecutwfc,
        global_ecutrho_ry=global_ecutrho,
    )
    queue_rows: list[dict[str, Any]] = []
    for row, structure, final_cif in sorted(
        structures, key=lambda item: int(item[0]["rank"])
    ):
        rank = int(row["rank"])
        candidate_id = row["candidate_id"]
        formula = structure.composition.reduced_formula
        job_dir = jobs_dir / f"{rank:02d}_{candidate_id}"
        job_dir.mkdir(parents=True, exist_ok=True)
        copied_cif = job_dir / "input_qe_relaxed.cif"
        shutil.copy2(final_cif, copied_cif)
        input_path = job_dir / "static-scf.in"
        output_path = job_dir / "static-scf.out"
        run_record = job_dir / "static_run.json"
        kgrid = _kpoint_grid(
            structure, float(config["static_kpoint_spacing_inv_angstrom"])
        )
        pseudo_map = {
            element.symbol: pseudo_entries[element.symbol]["filename"]
            for element in structure.composition.elements
        }
        pw_input = PWInput(
            structure,
            pseudo=pseudo_map,
            control={
                "calculation": "scf",
                "restart_mode": "from_scratch",
                "prefix": f"{candidate_id}_static",
                "pseudo_dir": "../../pseudos",
                "outdir": "./tmp",
                "disk_io": "low",
                "tstress": True,
                "tprnfor": True,
            },
            system={
                "input_dft": config["input_dft"],
                "ecutwfc": global_ecutwfc,
                "ecutrho": global_ecutrho,
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
            kpoints_grid=kgrid,
            kpoints_shift=(0, 0, 0),
        )
        pw_input.write_file(input_path)
        blockers = list(engine_blockers)
        job_status = (
            "runnable_not_started" if not blockers else "inputs_ready_engine_missing"
        )
        queue_rows.append({
            "rank": rank,
            "entry_id": row.get("entry_id") or f"candidate:{candidate_id}",
            "entry_role": row.get("entry_role") or "candidate",
            "source_id": row.get("source_id") or candidate_id,
            "candidate_id": candidate_id,
            "formula": formula,
            "job_status": job_status,
            "blockers": json.dumps(blockers),
            "final_relaxed_cif": str(final_cif),
            "final_relaxed_cif_sha256": _sha256(final_cif),
            "initial_source_cif_sha256": row.get("initial_source_cif_sha256") or "",
            "copied_cif": str(copied_cif),
            "kpoints_grid": "x".join(str(value) for value in kgrid),
            "qe_input": str(input_path),
            "qe_input_sha256": _sha256(input_path),
            "qe_output": str(output_path),
            "run_record": str(run_record),
            "parent_relax_settings_hash": expected_parent_hash,
            "parent_relax_input_settings_hash": expected_parent_input_hash,
            "static_settings_hash": static_hash,
            "calculation_started": False,
            "dft_static_validated": False,
        })

    queue_path = output_dir / "static_queue_manifest.csv"
    _write_csv(queue_path, queue_rows)
    queue_manifest_sha256 = _sha256(queue_path)
    overall_status = (
        "runnable_not_started" if executable_path else "inputs_ready_engine_missing"
    )
    preflight = {
        "workflow_version": WORKFLOW_VERSION,
        "status": overall_status,
        "calculation": "scf",
        "calculation_started": False,
        "dft_static_validated_count": 0,
        "relaxation_results": str(Path(relaxation_results_path).resolve()),
        "relax_preflight": str(relax_preflight_path),
        "source_inventory": relax_preflight.get("source_inventory"),
        "source_inventory_sha256": relax_preflight.get("source_inventory_sha256"),
        "config_source": str(config_path),
        "config": str(config_snapshot_path),
        "pseudo_manifest_source": str(pseudo_manifest_path),
        "pseudo_manifest": str(pseudo_manifest_snapshot_path),
        "candidate_count": len(queue_rows),
        "required_elements": required_elements,
        "global_ecutwfc_ry": global_ecutwfc,
        "global_ecutrho_ry": global_ecutrho,
        "parent_relax_settings_hash": expected_parent_hash,
        "parent_relax_input_settings_hash": expected_parent_input_hash,
        "static_settings_hash": static_hash,
        "bundled_pseudopotentials": [
            {
                "element": symbol,
                "filename": pseudo_entries[symbol]["filename"],
                "path": str(
                    bundled_pseudo_dir / pseudo_entries[symbol]["filename"]
                ),
                "sha256": pseudo_entries[symbol]["sha256"],
            }
            for symbol in required_elements
        ],
        "pw_executable_requested": str(requested_executable),
        "pw_executable_resolved": executable_path,
        "engine_blockers": engine_blockers,
        "resource_policy": config["resource_policy"],
        "queue_manifest": str(queue_path),
        "queue_manifest_sha256": queue_manifest_sha256,
        "scientific_limit": (
            "Static candidate energies alone are not formation energies or "
            "convex-hull validation. Consistent elemental and competing-phase "
            "reference calculations are still required."
        ),
    }
    preflight_path = output_dir / "static_preflight.json"
    preflight_path.write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    print(f"Static preparation status: {overall_status}")
    print(f"Candidates queued: {len(queue_rows)}")
    print(f"Queue:     {queue_path}")
    print(f"Preflight: {preflight_path}")
    print("No Quantum ESPRESSO calculation was started.")
    return preflight


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relaxation-results", type=Path, required=True)
    parser.add_argument("--relax-preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pw-executable")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    prepare_static_jobs(
        relaxation_results_path=args.relaxation_results,
        relax_preflight_path=args.relax_preflight,
        output_dir=args.output_dir,
        pw_executable=args.pw_executable,
    )
