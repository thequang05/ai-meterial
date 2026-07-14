"""Prepare (but never execute) Quantum ESPRESSO jobs for validated candidates.

The command always creates an auditable queue.  It emits ``vc-relax.in`` only
when every required pseudopotential passes official-metadata, license,
filename, cutoff, MD5/SHA-256, element, functional, and relativistic-header
checks. Missing ``pw.x`` is reported as a blocker; this module never launches
MPI or a DFT executable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
from pathlib import Path
from typing import Any

from pymatgen.core import Composition, Structure
from pymatgen.io.pwscf import PWInput

from prepare_sssp_manifest import _inspect_upf_header, _normalize_expected_md5


WORKFLOW_VERSION = "qe_pbe_candidate_relax_v1"
PSEUDO_MANIFEST_VERSION = "qe_pseudo_manifest_v1"
SAFE_JOB_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
SAFE_ENTRY_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}")


def _load_json(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.new("md5")
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _settings_hash(
    *,
    config_path: Path,
    pseudo_manifest_path: Path,
    global_ecutwfc_ry: float,
    global_ecutrho_ry: float,
) -> str:
    payload = {
        "workflow_version": WORKFLOW_VERSION,
        "config_sha256": _sha256(config_path),
        "pseudo_manifest_sha256": _sha256(pseudo_manifest_path),
        "global_ecutwfc_ry": global_ecutwfc_ry,
        "global_ecutrho_ry": global_ecutrho_ry,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _resolve_executable(value: str) -> str | None:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        candidate = candidate.resolve()
        return str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else None
    return shutil.which(value)


def _kpoint_grid(structure: Structure, spacing_inv_angstrom: float) -> tuple[int, int, int]:
    if spacing_inv_angstrom <= 0:
        raise ValueError("kpoint_spacing_inv_angstrom must be positive")
    # pymatgen's reciprocal_lattice includes 2*pi, so |b_i| / spacing gives
    # the number of intervals needed to keep adjacent k-points within spacing.
    return tuple(
        max(1, int(math.ceil(length / spacing_inv_angstrom)))
        for length in structure.lattice.reciprocal_lattice.abc
    )


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "qe_dft_config_v1":
        raise ValueError("Expected config schema_version=qe_dft_config_v1")
    if str(config.get("input_dft", "")).upper() != "PBE":
        raise ValueError("This workflow version is locked to PBE")
    if config.get("occupations") != "smearing" or config.get("smearing") != "mv":
        raise ValueError("This workflow version is locked to MV smearing")
    positive_fields = [
        "kpoint_spacing_inv_angstrom",
        "degauss_ry",
        "conv_thr",
        "etot_conv_thr_ry",
        "forc_conv_thr_ry_per_bohr",
        "press_conv_thr_kbar",
    ]
    for field in positive_fields:
        value = float(config.get(field, 0))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Config field must be positive: {field}")
    raw_nstep = config.get("nstep", 0)
    try:
        nstep = int(raw_nstep)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Config nstep must be a finite integer") from exc
    if isinstance(raw_nstep, float) and (
        not math.isfinite(raw_nstep) or raw_nstep != nstep
    ):
        raise ValueError("Config nstep must be a finite integer")
    if nstep < 1:
        raise ValueError("Config nstep must be at least 1")


def _candidate_queue(report: dict[str, Any], top_n: int) -> list[dict[str, Any]]:
    allowed_statuses = {
        "ml_structural_validation_complete",
        "reference_inventory_complete",
    }
    if report.get("status") not in allowed_statuses:
        raise ValueError(
            "Input report must be marked ml_structural_validation_complete or "
            "reference_inventory_complete"
        )
    raw_queue = report.get("recommended_dft_queue", [])
    if not isinstance(raw_queue, list):
        raise ValueError("recommended_dft_queue must be a list")
    selected = raw_queue[:top_n]
    if not selected:
        raise ValueError("Validation report contains no recommended DFT candidates")
    queue: list[dict[str, Any]] = []
    seen_ranks: set[int] = set()
    seen_candidate_ids: set[str] = set()
    seen_entry_ids: set[str] = set()
    for index, raw in enumerate(selected, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"DFT queue row {index} must be an object")
        candidate = dict(raw)
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if not SAFE_JOB_ID_RE.fullmatch(candidate_id):
            raise ValueError(f"Unsafe candidate_id at DFT queue row {index}")
        raw_rank = candidate.get("rank")
        try:
            rank = int(raw_rank)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid rank at DFT queue row {index}") from exc
        if (
            isinstance(raw_rank, bool)
            or rank < 1
            or isinstance(raw_rank, float) and raw_rank != rank
        ):
            raise ValueError(f"Invalid rank at DFT queue row {index}")
        entry_role = str(candidate.get("entry_role") or "candidate")
        entry_id = str(
            candidate.get("entry_id") or f"{entry_role}:{candidate_id}"
        ).strip()
        if not SAFE_ENTRY_ID_RE.fullmatch(entry_id):
            raise ValueError(f"Unsafe entry_id at DFT queue row {index}")
        if rank in seen_ranks:
            raise ValueError(f"Duplicate DFT queue rank: {rank}")
        if candidate_id in seen_candidate_ids:
            raise ValueError(f"Duplicate DFT candidate_id: {candidate_id}")
        if entry_id in seen_entry_ids:
            raise ValueError(f"Duplicate DFT entry_id: {entry_id}")
        seen_ranks.add(rank)
        seen_candidate_ids.add(candidate_id)
        seen_entry_ids.add(entry_id)
        candidate["rank"] = rank
        candidate["candidate_id"] = candidate_id
        candidate["entry_role"] = entry_role
        candidate["entry_id"] = entry_id
        queue.append(candidate)
    return queue


def _validate_pseudopotentials(
    *,
    required_elements: list[str],
    pseudo_manifest_path: Path | None,
    pseudo_dir: Path | None,
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, Any] | None]:
    blockers: list[str] = []
    if pseudo_manifest_path is None:
        return {}, ["pseudo_manifest_not_provided"], None
    if pseudo_dir is None:
        return {}, ["pseudo_dir_not_provided"], None

    manifest = _load_json(pseudo_manifest_path)
    if manifest.get("schema_version") != PSEUDO_MANIFEST_VERSION:
        blockers.append("pseudo_manifest_schema_mismatch")
    if str(manifest.get("functional", "")).upper() != "PBE":
        blockers.append("pseudo_functional_not_pbe")
    if str(manifest.get("relativistic", "")).lower() not in {
        "scalar_relativistic",
        "scalar-relativistic",
    }:
        blockers.append("pseudo_relativistic_policy_mismatch")
    if manifest.get("licenses_acknowledged_by_user") is not True:
        blockers.append("pseudo_original_licenses_not_acknowledged")
    if str(manifest.get("library") or "").strip() != "SSSP PBE Precision":
        blockers.append("pseudo_library_not_sssp_pbe_precision")
    if not str(manifest.get("library_version") or "").strip():
        blockers.append("pseudo_library_version_missing")

    source_metadata: dict[str, Any] | None = None
    source_metadata_text = str(manifest.get("source_metadata") or "").strip()
    source_metadata_sha = str(
        manifest.get("source_metadata_sha256") or ""
    ).strip().lower()
    if not source_metadata_text:
        blockers.append("pseudo_source_metadata_missing")
    elif not re.fullmatch(r"[0-9a-f]{64}", source_metadata_sha):
        blockers.append("pseudo_source_metadata_sha256_invalid")
    else:
        source_metadata_path = Path(source_metadata_text).expanduser().resolve()
        if not source_metadata_path.is_file():
            blockers.append("pseudo_source_metadata_file_missing")
        elif _sha256(source_metadata_path) != source_metadata_sha:
            blockers.append("pseudo_source_metadata_sha256_mismatch")
        else:
            loaded_metadata = _load_json(source_metadata_path)
            if not isinstance(loaded_metadata, dict):
                blockers.append("pseudo_source_metadata_not_object")
            else:
                source_metadata = loaded_metadata

    pseudo_dir = Path(pseudo_dir).resolve()
    entries: dict[str, dict[str, Any]] = {}
    raw_entries = manifest.get("elements", {})
    if not isinstance(raw_entries, dict):
        blockers.append("pseudo_elements_metadata_not_object")
        raw_entries = {}
    selected_filenames: set[str] = set()
    for symbol in required_elements:
        raw = raw_entries.get(symbol)
        if not isinstance(raw, dict):
            blockers.append(f"pseudo_metadata_missing:{symbol}")
            continue
        filename = str(raw.get("filename") or "").strip()
        expected_sha = str(raw.get("sha256") or "").strip().lower()
        try:
            ecutwfc = float(raw["ecutwfc_ry"])
            ecutrho = float(raw["ecutrho_ry"])
        except (KeyError, TypeError, ValueError):
            blockers.append(f"pseudo_cutoff_invalid:{symbol}")
            continue
        if (
            not math.isfinite(ecutwfc)
            or not math.isfinite(ecutrho)
            or ecutwfc <= 0
            or ecutrho < ecutwfc
        ):
            blockers.append(f"pseudo_cutoff_invalid:{symbol}")
        if not filename:
            blockers.append(f"pseudo_filename_missing:{symbol}")
            continue
        filename_path = Path(filename)
        if filename_path.is_absolute() or filename_path.parent != Path("."):
            blockers.append(f"pseudo_filename_unsafe:{symbol}:{filename}")
            continue
        if filename in selected_filenames:
            blockers.append(f"pseudo_filename_duplicate:{symbol}:{filename}")
            continue
        selected_filenames.add(filename)
        source = pseudo_dir / filename
        if not source.is_file():
            blockers.append(f"pseudo_file_missing:{symbol}:{filename}")
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            blockers.append(f"pseudo_sha256_missing:{symbol}")
            continue
        actual_sha = _sha256(source)
        if actual_sha != expected_sha:
            blockers.append(f"pseudo_sha256_mismatch:{symbol}:{filename}")
            continue
        expected_md5 = str(raw.get("original_md5") or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{32}", expected_md5):
            blockers.append(f"pseudo_original_md5_missing:{symbol}")
            continue
        actual_md5 = _md5(source)
        if actual_md5 != expected_md5:
            blockers.append(f"pseudo_original_md5_mismatch:{symbol}:{filename}")
            continue
        try:
            actual_header = _inspect_upf_header(source, symbol)
        except ValueError as exc:
            blockers.append(f"pseudo_upf_header_invalid:{symbol}:{exc}")
            continue
        attested_header = {
            "element": str(raw.get("upf_header_element") or "").strip(),
            "functional": str(raw.get("upf_header_functional") or "").strip(),
            "relativistic": str(raw.get("upf_header_relativistic") or "").strip(),
        }
        if attested_header != actual_header:
            blockers.append(f"pseudo_upf_header_attestation_mismatch:{symbol}")
            continue
        if source_metadata is not None:
            official = source_metadata.get(symbol)
            if not isinstance(official, dict):
                blockers.append(f"pseudo_official_metadata_missing:{symbol}")
                continue
            official_md5 = _normalize_expected_md5(
                official.get("md5") or official.get("checksum")
            )
            if (
                str(official.get("filename") or "").strip() != filename
                or official_md5 != expected_md5
            ):
                blockers.append(f"pseudo_official_identity_mismatch:{symbol}")
                continue
            try:
                official_ecutwfc = float(official["cutoff"])
                if "dual" in official:
                    official_ecutrho = official_ecutwfc * float(official["dual"])
                else:
                    official_ecutrho = float(official["ecutrho"])
            except (KeyError, TypeError, ValueError):
                blockers.append(f"pseudo_official_cutoff_invalid:{symbol}")
                continue
            if not (
                math.isclose(ecutwfc, official_ecutwfc, rel_tol=0, abs_tol=1e-10)
                and math.isclose(
                    ecutrho, official_ecutrho, rel_tol=0, abs_tol=1e-8
                )
            ):
                blockers.append(f"pseudo_official_cutoff_mismatch:{symbol}")
                continue
        entries[symbol] = {
            "filename": filename,
            "source_path": str(source),
            "sha256": actual_sha,
            "ecutwfc_ry": ecutwfc,
            "ecutrho_ry": ecutrho,
        }
    return entries, blockers, manifest


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    preferred = [
        "rank", "entry_id", "entry_role", "source_id", "candidate_id",
        "formula", "job_status", "blockers",
        "required_elements", "kpoints_grid", "source_cif", "source_cif_sha256",
        "copied_cif", "qe_input", "qe_input_sha256", "qe_output",
        "run_record", "job_record",
    ]
    extras = sorted({key for row in rows for key in row} - set(preferred))
    fields = preferred + extras
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def prepare_jobs(
    *,
    validation_report_path: Path,
    config_path: Path,
    output_dir: Path,
    top_n: int = 5,
    pseudo_manifest_path: Path | None = None,
    pseudo_dir: Path | None = None,
    pw_executable: str = "pw.x",
) -> dict[str, Any]:
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    report = _load_json(validation_report_path)
    config = _load_json(config_path)
    _validate_config(config)
    queue = _candidate_queue(report, top_n)

    output_dir = Path(output_dir).resolve()
    jobs_dir = output_dir / "jobs"
    bundled_pseudo_dir = output_dir / "pseudos"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    config_source_path = Path(config_path).resolve()
    config_snapshot_path = output_dir / "workflow_config_snapshot.json"
    config_snapshot_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    structures: list[tuple[dict[str, Any], Structure, Path]] = []
    required_elements_set: set[str] = set()
    for candidate in queue:
        source_cif = Path(candidate["relaxed_cif"]).expanduser().resolve()
        if not source_cif.is_file():
            raise FileNotFoundError(source_cif)
        structure = Structure.from_file(source_cif)
        expected = Composition(candidate["formula"]).reduced_formula
        actual = structure.composition.reduced_formula
        if actual != expected:
            raise ValueError(
                f"Candidate formula mismatch for {candidate['candidate_id']}: "
                f"expected={expected}, actual={actual}"
            )
        required_elements_set.update(element.symbol for element in structure.composition.elements)
        structures.append((candidate, structure, source_cif))

    required_elements = sorted(
        required_elements_set,
        key=lambda symbol: Composition(symbol).elements[0].Z,
    )
    pseudo_entries, pseudo_blockers, pseudo_manifest = _validate_pseudopotentials(
        required_elements=required_elements,
        pseudo_manifest_path=pseudo_manifest_path,
        pseudo_dir=pseudo_dir,
    )
    executable_path = _resolve_executable(pw_executable)
    engine_blockers = [] if executable_path else [f"pw_executable_not_found:{pw_executable}"]
    inputs_ready = not pseudo_blockers

    global_ecutwfc = (
        max(entry["ecutwfc_ry"] for entry in pseudo_entries.values())
        if inputs_ready else None
    )
    global_ecutrho = (
        max(entry["ecutrho_ry"] for entry in pseudo_entries.values())
        if inputs_ready else None
    )
    if inputs_ready:
        bundled_pseudo_dir.mkdir(parents=True, exist_ok=True)
        for entry in pseudo_entries.values():
            shutil.copy2(entry["source_path"], bundled_pseudo_dir / entry["filename"])
    pseudo_manifest_snapshot_path: Path | None = None
    if pseudo_manifest_path is not None and pseudo_manifest is not None:
        pseudo_manifest_snapshot_path = output_dir / "pseudo_manifest_snapshot.json"
        shutil.copy2(Path(pseudo_manifest_path).resolve(), pseudo_manifest_snapshot_path)

    relax_settings_hash = (
        _settings_hash(
            config_path=config_snapshot_path,
            pseudo_manifest_path=pseudo_manifest_snapshot_path,
            global_ecutwfc_ry=float(global_ecutwfc),
            global_ecutrho_ry=float(global_ecutrho),
        )
        if inputs_ready and pseudo_manifest_snapshot_path is not None
        else None
    )

    rows: list[dict[str, Any]] = []
    for candidate, structure, source_cif in structures:
        rank = int(candidate["rank"])
        candidate_id = str(candidate["candidate_id"])
        entry_role = str(candidate.get("entry_role") or "candidate")
        if entry_role not in {"candidate", "reference"}:
            raise ValueError(f"Unsupported entry_role for {candidate_id}: {entry_role}")
        entry_id = str(
            candidate.get("entry_id") or f"{entry_role}:{candidate_id}"
        )
        source_id = str(candidate.get("source_id") or candidate_id)
        job_dir = jobs_dir / f"{rank:02d}_{candidate_id}"
        job_dir.mkdir(parents=True, exist_ok=True)
        copied_cif = job_dir / "input_chgnet_relaxed.cif"
        shutil.copy2(source_cif, copied_cif)
        qe_input_path = job_dir / "vc-relax.in"
        qe_output_path = job_dir / "vc-relax.out"
        run_record_path = job_dir / "job_run.json"
        kgrid = _kpoint_grid(
            structure, float(config["kpoint_spacing_inv_angstrom"])
        )

        blockers = list(pseudo_blockers) + list(engine_blockers)
        if inputs_ready:
            pseudo_map = {
                symbol: pseudo_entries[symbol]["filename"]
                for symbol in sorted(
                    {element.symbol for element in structure.composition.elements}
                )
            }
            pw_input = PWInput(
                structure,
                pseudo=pseudo_map,
                control={
                    "calculation": "vc-relax",
                    "restart_mode": "from_scratch",
                    "prefix": candidate_id,
                    "pseudo_dir": "../../pseudos",
                    "outdir": "./tmp",
                    "disk_io": "low",
                    "tstress": True,
                    "tprnfor": True,
                    "nstep": int(config["nstep"]),
                    "etot_conv_thr": float(config["etot_conv_thr_ry"]),
                    "forc_conv_thr": float(config["forc_conv_thr_ry_per_bohr"]),
                },
                system={
                    "input_dft": config["input_dft"],
                    "ecutwfc": global_ecutwfc,
                    "ecutrho": global_ecutrho,
                    "occupations": config["occupations"],
                    "smearing": config["smearing"],
                    "degauss": float(config["degauss_ry"]),
                },
                electrons={
                    "conv_thr": float(config["conv_thr"]),
                    "electron_maxstep": int(config["electron_maxstep"]),
                    "mixing_beta": float(config["mixing_beta"]),
                    "diagonalization": config["diagonalization"],
                },
                ions={"ion_dynamics": "bfgs"},
                cell={
                    "cell_dynamics": "bfgs",
                    "press": 0.0,
                    "press_conv_thr": float(config["press_conv_thr_kbar"]),
                    "cell_dofree": config["cell_dofree"],
                },
                kpoints_mode="automatic",
                kpoints_grid=kgrid,
                kpoints_shift=(0, 0, 0),
            )
            pw_input.write_file(qe_input_path)
        qe_input_sha256 = _sha256(qe_input_path) if qe_input_path.is_file() else ""

        job_status = (
            "runnable_not_started"
            if inputs_ready and executable_path
            else "inputs_ready_engine_missing"
            if inputs_ready
            else "planned_waiting_for_pseudopotentials"
        )
        job_record = {
            "workflow_version": WORKFLOW_VERSION,
            "rank": rank,
            "entry_id": entry_id,
            "entry_role": entry_role,
            "source_id": source_id,
            "candidate_id": candidate_id,
            "formula": candidate["formula"],
            "job_status": job_status,
            "blockers": blockers,
            "source_cif": str(source_cif),
            "source_cif_sha256": _sha256(source_cif),
            "copied_cif": str(copied_cif),
            "qe_input": str(qe_input_path) if inputs_ready else "",
            "qe_input_sha256": qe_input_sha256,
            "qe_output": str(qe_output_path),
            "run_record": str(run_record_path),
            "kpoints_grid": list(kgrid),
            "ecutwfc_ry": global_ecutwfc,
            "ecutrho_ry": global_ecutrho,
            "calculation_started": False,
            "dft_validated": False,
            "relax_settings_hash": relax_settings_hash,
            "scientific_limit": (
                "Prepared input only. No Quantum ESPRESSO calculation has been run."
            ),
        }
        job_record_path = job_dir / "job_plan.json"
        job_record_path.write_text(json.dumps(job_record, indent=2), encoding="utf-8")
        rows.append({
            **job_record,
            "blockers": json.dumps(blockers),
            "required_elements": json.dumps(
                sorted(element.symbol for element in structure.composition.elements)
            ),
            "kpoints_grid": "x".join(str(value) for value in kgrid),
            "job_record": str(job_record_path),
        })

    queue_manifest_path = output_dir / "dft_queue_manifest.csv"
    _write_csv(queue_manifest_path, rows)
    queue_manifest_sha256 = _sha256(queue_manifest_path)
    overall_status = (
        "runnable_not_started"
        if inputs_ready and executable_path
        else "inputs_ready_engine_missing"
        if inputs_ready
        else "blocked_missing_pseudopotentials"
    )
    preflight = {
        "workflow_version": WORKFLOW_VERSION,
        "status": overall_status,
        "calculation_started": False,
        "dft_validated_count": 0,
        "validation_report": str(Path(validation_report_path).resolve()),
        "source_inventory": report.get("source_inventory"),
        "source_inventory_sha256": report.get("source_inventory_sha256"),
        "config_source": str(config_source_path),
        "config": str(config_snapshot_path),
        "candidate_count": len(rows),
        "required_elements": required_elements,
        "pseudo_manifest_source": (
            str(Path(pseudo_manifest_path).resolve()) if pseudo_manifest_path else None
        ),
        "pseudo_manifest": (
            str(pseudo_manifest_snapshot_path) if pseudo_manifest_snapshot_path else None
        ),
        "pseudo_library": pseudo_manifest.get("library") if pseudo_manifest else None,
        "pseudo_library_version": (
            pseudo_manifest.get("library_version") if pseudo_manifest else None
        ),
        "pseudo_blockers": pseudo_blockers,
        "pw_executable_requested": pw_executable,
        "pw_executable_resolved": executable_path,
        "engine_blockers": engine_blockers,
        "global_ecutwfc_ry": global_ecutwfc,
        "global_ecutrho_ry": global_ecutrho,
        "relax_settings_hash": relax_settings_hash,
        "relax_input_settings_hash": relax_settings_hash,
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
            if symbol in pseudo_entries
        ],
        "machine": {
            "architecture": platform.machine(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
        "resource_policy": config["resource_policy"],
        "queue_manifest": str(queue_manifest_path),
        "queue_manifest_sha256": queue_manifest_sha256,
        "scientific_limit": (
            "This artifact only prepares candidate relaxation inputs. Formation "
            "energies and convex-hull stability require completed, converged DFT "
            "calculations for candidates and all relevant reference phases using "
            "one consistent setup."
        ),
    }
    preflight_path = output_dir / "dft_preflight.json"
    preflight_path.write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    print(f"DFT preparation status: {overall_status}")
    print(f"Candidates queued: {len(rows)}")
    print(f"Required elements: {','.join(required_elements)}")
    if pseudo_blockers:
        print(f"Pseudopotential blockers: {len(pseudo_blockers)}")
        for blocker in pseudo_blockers:
            print(f"  - {blocker}")
    if engine_blockers:
        print(f"Engine blocker: {engine_blockers[0]}")
    print(f"Queue:     {queue_manifest_path}")
    print(f"Preflight: {preflight_path}")
    print("No DFT calculation was started.")
    return preflight


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--pseudo-manifest", type=Path)
    parser.add_argument("--pseudo-dir", type=Path)
    parser.add_argument("--pw-executable", default="pw.x")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    prepare_jobs(
        validation_report_path=args.validation_report,
        config_path=args.config,
        output_dir=args.output_dir,
        top_n=args.top_n,
        pseudo_manifest_path=args.pseudo_manifest,
        pseudo_dir=args.pseudo_dir,
        pw_executable=args.pw_executable,
    )
