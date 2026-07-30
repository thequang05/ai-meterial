"""Prepare (but never execute) consistent Quantum ESPRESSO static-SCF jobs.

Only candidates that passed ``collect_qe_relaxations.py`` are accepted.  The
same PBE pseudopotentials, convergence certificate, global cutoffs, QE binary
lineage, and relaxed-result provenance are verified again.  The effective
static grid is no coarser than both the locked config and certificate. This
module writes inputs and an auditable queue; it never launches ``pw.x``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from pymatgen.core import Composition, Structure
from pymatgen.io.pwscf import PWInput

from prepare_qe_jobs import (
    _kpoint_grid,
    _ensure_fresh_output_dir,
    _qe_scratch_outdir,
    _qe_scratch_campaign_namespace,
    _prepare_qe_scratch_campaign,
    _resolve_executable,
    _resolve_scratch_root,
    _sha256,
    _validate_config,
    _validate_pseudopotentials,
)
from qe_convergence_certificate import verify_convergence_certificate
from qe_execution_provenance import require_same_execution_provenance


WORKFLOW_VERSION = "qe_pbe_candidate_static_v1"
RELAX_COLLECTOR_VERSION = "qe_relax_collector_v1"


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
        "qe_input_sha256", "run_record", "qe_scratch_outdir",
        "parent_relax_settings_hash",
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
    recorded = str(
        preflight.get("relax_settings_hash")
        or preflight.get("relax_input_settings_hash")
        or ""
    )
    if recorded != computed:
        raise ValueError(
            "Relax preflight settings hash no longer matches its locked artifacts"
        )
    return computed


def _static_settings_hash(
    *,
    parent_relax_settings_hash: str,
    config_path: Path,
    pseudo_manifest_path: Path,
    global_ecutwfc_ry: float,
    global_ecutrho_ry: float,
    effective_static_kpoint_spacing_inv_angstrom: float,
    convergence_certificate_payload_sha256: str,
) -> str:
    payload = {
        "workflow_version": WORKFLOW_VERSION,
        "parent_relax_settings_hash": parent_relax_settings_hash,
        "config_sha256": _sha256(config_path),
        "pseudo_manifest_sha256": _sha256(pseudo_manifest_path),
        "global_ecutwfc_ry": global_ecutwfc_ry,
        "global_ecutrho_ry": global_ecutrho_ry,
        "effective_static_kpoint_spacing_inv_angstrom": (
            effective_static_kpoint_spacing_inv_angstrom
        ),
        "convergence_certificate_payload_sha256": (
            convergence_certificate_payload_sha256
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "").lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} is not a valid SHA-256 digest")
    return digest


def _recorded_path(value: Any, *, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is missing")
    return Path(text).expanduser().resolve()


def _row_identity(row: dict[str, str]) -> tuple[str, ...]:
    try:
        rank = str(int(row["rank"]))
        formula = Composition(row["formula"]).reduced_formula
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid relaxation inventory row: {row}") from exc
    return (
        rank,
        str(row.get("entry_id") or ""),
        str(row.get("entry_role") or ""),
        str(row.get("source_id") or ""),
        str(row.get("candidate_id") or ""),
        formula,
    )


def _verify_relaxation_collection(
    *,
    relaxation_results_path: Path,
    relaxation_summary_path: Path,
    relax_preflight_path: Path,
    relax_preflight: dict[str, Any],
    certificate: dict[str, Any],
    expected_relax_settings_hash: str,
) -> list[dict[str, str]]:
    """Verify the collector receipt and its complete immutable input chain."""

    relaxation_results_path = Path(relaxation_results_path).resolve()
    relaxation_summary_path = Path(relaxation_summary_path).resolve()
    summary = _load_json(relaxation_summary_path)
    if summary.get("collector_version") != RELAX_COLLECTOR_VERSION:
        raise ValueError(
            "Unsupported or missing relaxation collector version: "
            f"{summary.get('collector_version')}"
        )

    bound_results = _recorded_path(
        summary.get("results_manifest"), label="summary results_manifest"
    )
    if bound_results != relaxation_results_path:
        raise ValueError("Relaxation summary points to a different results manifest")
    expected_results_sha = _require_sha256(
        summary.get("results_manifest_sha256"),
        label="summary results_manifest_sha256",
    )
    if (
        not relaxation_results_path.is_file()
        or _sha256(relaxation_results_path) != expected_results_sha
    ):
        raise ValueError("Relaxation results manifest hash mismatch")

    bound_preflight = _recorded_path(
        summary.get("source_preflight"), label="summary source_preflight"
    )
    if bound_preflight != relax_preflight_path:
        raise ValueError("Relaxation summary points to a different relax preflight")
    expected_preflight_sha = _require_sha256(
        summary.get("source_preflight_sha256"),
        label="summary source_preflight_sha256",
    )
    if _sha256(relax_preflight_path) != expected_preflight_sha:
        raise ValueError("Relax preflight hash differs from the collector receipt")

    queue_path = _recorded_path(
        relax_preflight.get("queue_manifest"), label="relax queue_manifest"
    )
    bound_queue = _recorded_path(
        summary.get("source_queue_manifest"),
        label="summary source_queue_manifest",
    )
    if bound_queue != queue_path:
        raise ValueError("Relaxation summary points to a different relax queue")
    preflight_queue_sha = _require_sha256(
        relax_preflight.get("queue_manifest_sha256"),
        label="relax preflight queue_manifest_sha256",
    )
    summary_queue_sha = _require_sha256(
        summary.get("source_queue_manifest_sha256"),
        label="summary source_queue_manifest_sha256",
    )
    if (
        summary_queue_sha != preflight_queue_sha
        or not queue_path.is_file()
        or _sha256(queue_path) != preflight_queue_sha
    ):
        raise ValueError("Relax queue hash differs from its preflight/collector receipt")

    if summary.get("relax_input_settings_hash") != expected_relax_settings_hash:
        raise ValueError("Relaxation summary settings hash mismatch")
    expected_certificate_path = Path(certificate["certificate_path"]).resolve()
    if _recorded_path(
        summary.get("convergence_certificate"),
        label="summary convergence_certificate",
    ) != expected_certificate_path:
        raise ValueError(
            "Relaxation summary points to a different convergence certificate"
        )
    for key, verified_key in (
        ("convergence_certificate_sha256", "certificate_sha256"),
        ("convergence_certificate_id", "certificate_id"),
        (
            "convergence_certificate_payload_sha256",
            "certificate_payload_sha256",
        ),
    ):
        if summary.get(key) != certificate[verified_key]:
            raise ValueError(f"Relaxation summary certificate mismatch: {key}")
    require_same_execution_provenance(
        summary.get("execution_provenance"),
        certificate["execution_provenance"],
        label="convergence certificate",
    )

    all_rows = _read_csv(relaxation_results_path)
    queue_rows = _read_csv(queue_path)
    if not all_rows:
        raise ValueError("Relaxation results manifest is empty")
    allowed_statuses = {"dft_relax_converged", "dft_relax_failed_gate"}
    observed_statuses = Counter(row.get("relax_gate_status", "") for row in all_rows)
    if set(observed_statuses) - allowed_statuses:
        raise ValueError(
            "Relaxation results contain unknown gate statuses: "
            f"{sorted(set(observed_statuses) - allowed_statuses)}"
        )
    if int(summary.get("candidate_count", -1)) != len(all_rows):
        raise ValueError("Relaxation summary candidate_count mismatch")
    if int(relax_preflight.get("candidate_count", -1)) != len(queue_rows):
        raise ValueError("Relax preflight candidate_count mismatch")
    recorded_statuses = {
        str(key): int(value)
        for key, value in dict(summary.get("status_counts") or {}).items()
    }
    if recorded_statuses != dict(observed_statuses):
        raise ValueError("Relaxation summary status_counts mismatch")
    converged_count = observed_statuses.get("dft_relax_converged", 0)
    if int(summary.get("dft_relax_converged_count", -1)) != converged_count:
        raise ValueError("Relaxation summary converged-count mismatch")

    result_identities = [_row_identity(row) for row in all_rows]
    queue_identities = [_row_identity(row) for row in queue_rows]
    if len(set(result_identities)) != len(result_identities):
        raise ValueError("Relaxation results contain duplicate inventory entries")
    if len(set(queue_identities)) != len(queue_identities):
        raise ValueError("Relax queue contains duplicate inventory entries")
    if set(result_identities) != set(queue_identities):
        raise ValueError("Relaxation results inventory differs from the locked queue")
    if {
        row.get("relax_input_settings_hash", "") for row in all_rows
    } != {expected_relax_settings_hash}:
        raise ValueError("Relaxation rows use settings outside the locked preflight")

    converged_hashes = sorted({
        row.get("settings_hash", "")
        for row in all_rows
        if row.get("relax_gate_status") == "dft_relax_converged"
    })
    if summary.get("settings_hashes") != converged_hashes:
        raise ValueError("Relaxation summary effective settings hashes mismatch")
    expected_single_hash = (
        converged_hashes[0] if len(converged_hashes) == 1 else None
    )
    if summary.get("settings_hash") != expected_single_hash:
        raise ValueError("Relaxation summary effective settings hash mismatch")
    return all_rows


def prepare_static_jobs(
    *,
    relaxation_results_path: Path,
    relaxation_summary_path: Path,
    relax_preflight_path: Path,
    output_dir: Path,
    pw_executable: str | None = None,
    scratch_root: Path | None = None,
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
    if relax_preflight.get("production_settings_certified") is not True:
        raise ValueError("Relax preflight is not convergence-certified")

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

    pseudo_manifest_path = Path(relax_preflight["pseudo_manifest"]).resolve()
    requested_executable = (
        pw_executable
        or relax_preflight.get("pw_executable_resolved")
        or relax_preflight.get("pw_executable_requested")
        or "pw.x"
    )
    executable_path = _resolve_executable(str(requested_executable))
    certificate = verify_convergence_certificate(
        Path(str(relax_preflight.get("convergence_certificate") or "")),
        required_elements=relax_preflight.get("required_elements") or [],
        config_path=config_path,
        pseudo_manifest_path=pseudo_manifest_path,
        pw_executable_path=(Path(executable_path) if executable_path else None),
    )
    if (
        relax_preflight.get("convergence_certificate_sha256")
        != certificate["certificate_sha256"]
        or relax_preflight.get("convergence_certificate_payload_sha256")
        != certificate["certificate_payload_sha256"]
    ):
        raise ValueError("Relax preflight convergence-certificate lineage mismatch")

    expected_parent_input_hash = _relax_settings_hash(relax_preflight)
    all_rows = _verify_relaxation_collection(
        relaxation_results_path=relaxation_results_path,
        relaxation_summary_path=relaxation_summary_path,
        relax_preflight_path=relax_preflight_path,
        relax_preflight=relax_preflight,
        certificate=certificate,
        expected_relax_settings_hash=expected_parent_input_hash,
    )
    converged = [
        row for row in all_rows
        if row.get("relax_gate_status") == "dft_relax_converged"
    ]
    if not converged:
        raise ValueError("No dft_relax_converged candidates were found")

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
    for row in converged:
        if (
            row.get("convergence_certificate_payload_sha256")
            != certificate["certificate_payload_sha256"]
            or row.get("qe_program_version") != certificate["qe_program_version"]
            or row.get("pw_executable_sha256") != certificate["pw_executable_sha256"]
        ):
            raise ValueError(
                "Relaxation result does not match the convergence certificate"
            )
        try:
            require_same_execution_provenance(
                json.loads(str(row.get("execution_provenance") or "")),
                certificate["execution_provenance"],
                label="convergence certificate",
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                "Relaxation result execution provenance does not match the "
                "convergence certificate"
            ) from exc

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
    production = certificate["production_settings"]
    if global_ecutwfc + 1e-10 < float(production["ecutwfc_ry"]):
        raise ValueError("Relax ecutwfc is below the certified production setting")
    if global_ecutrho + 1e-8 < float(production["ecutrho_ry"]):
        raise ValueError("Relax ecutrho is below the certified production setting")
    effective_static_spacing = min(
        float(config["static_kpoint_spacing_inv_angstrom"]),
        float(production["kpoint_spacing_inv_angstrom"]),
    )
    engine_blockers = (
        [] if executable_path else [f"pw_executable_not_found:{requested_executable}"]
    )

    output_dir = _ensure_fresh_output_dir(output_dir)
    jobs_dir = output_dir / "jobs"
    bundled_pseudo_dir = output_dir / "pseudos"
    resolved_scratch_root = _resolve_scratch_root(scratch_root)
    scratch_campaign_namespace = _qe_scratch_campaign_namespace(
        workflow_version=WORKFLOW_VERSION, output_dir=output_dir
    )
    scratch_campaign_dir = _prepare_qe_scratch_campaign(
        scratch_root=resolved_scratch_root,
        campaign_namespace=scratch_campaign_namespace,
    )
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
        effective_static_kpoint_spacing_inv_angstrom=effective_static_spacing,
        convergence_certificate_payload_sha256=certificate[
            "certificate_payload_sha256"
        ],
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
        qe_scratch_outdir = _qe_scratch_outdir(
            scratch_root=resolved_scratch_root,
            campaign_namespace=scratch_campaign_namespace,
            job_id=f"{rank:02d}_{candidate_id}",
        )
        copied_cif = job_dir / "input_qe_relaxed.cif"
        shutil.copy2(final_cif, copied_cif)
        input_path = job_dir / "static-scf.in"
        output_path = job_dir / "static-scf.out"
        run_record = job_dir / "static_run.json"
        kgrid = _kpoint_grid(structure, effective_static_spacing)
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
                "outdir": qe_scratch_outdir,
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
            "qe_scratch_outdir": qe_scratch_outdir,
            "parent_relax_settings_hash": expected_parent_hash,
            "parent_relax_input_settings_hash": expected_parent_input_hash,
            "static_settings_hash": static_hash,
            "effective_static_kpoint_spacing_inv_angstrom": effective_static_spacing,
            "convergence_certificate_id": certificate["certificate_id"],
            "convergence_certificate_payload_sha256": certificate[
                "certificate_payload_sha256"
            ],
            "certified_execution_provenance": certificate[
                "execution_provenance"
            ],
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
        "relaxation_results_sha256": _sha256(Path(relaxation_results_path).resolve()),
        "relaxation_summary": str(Path(relaxation_summary_path).resolve()),
        "relaxation_summary_sha256": _sha256(Path(relaxation_summary_path).resolve()),
        "relax_preflight": str(relax_preflight_path),
        "relax_preflight_sha256": _sha256(relax_preflight_path),
        "relax_queue_manifest": str(
            Path(relax_preflight["queue_manifest"]).resolve()
        ),
        "relax_queue_manifest_sha256": relax_preflight[
            "queue_manifest_sha256"
        ],
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
        "effective_static_kpoint_spacing_inv_angstrom": effective_static_spacing,
        "production_settings_certified": True,
        "convergence_certificate": certificate["certificate_path"],
        "convergence_certificate_sha256": certificate["certificate_sha256"],
        "convergence_certificate_id": certificate["certificate_id"],
        "convergence_certificate_payload_sha256": certificate[
            "certificate_payload_sha256"
        ],
        "convergence_qe_program_version": certificate["qe_program_version"],
        "convergence_pw_executable_sha256": certificate[
            "pw_executable_sha256"
        ],
        "certified_execution_provenance": certificate[
            "execution_provenance"
        ],
        "confirmed_selected_settings": certificate["selected_settings"],
        "certified_production_settings": certificate["production_settings"],
        "production_settings_scope": certificate["production_settings_scope"],
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
    parser.add_argument("--relaxation-summary", type=Path, required=True)
    parser.add_argument("--relax-preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pw-executable")
    parser.add_argument(
        "--scratch-root", type=Path,
        help=(
            "Optional shared QE scratch root. Each static job uses a distinct "
            "<workflow>/<campaign>/<job-id> directory; default is local ./tmp."
        ),
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    prepare_static_jobs(
        relaxation_results_path=args.relaxation_results,
        relaxation_summary_path=args.relaxation_summary,
        relax_preflight_path=args.relax_preflight,
        output_dir=args.output_dir,
        pw_executable=args.pw_executable,
        scratch_root=args.scratch_root,
    )
