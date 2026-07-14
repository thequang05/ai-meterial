"""Collect sparse QE convergence sweeps and make a provisional selection.

The collector fails closed on incomplete outputs or broken provenance.  It
compares energy, every atomic-force component, and every stress component to
the shared high-cutoff/dense-grid anchor.  A level passes only when it and the
entire higher/denser tail remains within the locked protocol tolerances.

The resulting selection is provisional.  This module deliberately does not
emit a production convergence certificate; an independent confirmation run at
the selected cutoff/k-grid combination is still required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_qe_convergence_jobs import (
    WORKFLOW_VERSION,
    convergence_settings_hash,
    validate_protocol,
)
from qe_output import summarize_qe_output


COLLECTOR_VERSION = "qe_convergence_collector_v1"
RY_TO_EV = 13.605693122994
BOHR_TO_ANGSTROM = 0.529177210903
RY_PER_BOHR_TO_EV_PER_ANGSTROM = RY_TO_EV / BOHR_TO_ANGSTROM
NUMBER_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
)
HASH_RE = re.compile(r"[0-9a-f]{64}")


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(value: str) -> float:
    result = float(value.replace("D", "E").replace("d", "e"))
    if not math.isfinite(result):
        raise ValueError("Non-finite QE numeric value")
    return result


def _parse_force_components(path: Path, num_atoms: int) -> list[list[float]]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    header_indices = [
        index for index, line in enumerate(lines)
        if re.search(r"forces\s+acting\s+on\s+atoms", line, re.I)
    ]
    if not header_indices:
        raise ValueError("No QE atomic-force block header was found")
    start = header_indices[-1] + 1
    vectors: list[list[float]] = []
    for line in lines[start:]:
        if re.search(r"forces\s+acting\s+on\s+atoms", line, re.I):
            vectors = []
            continue
        if "force =" in line.lower() and "atom" in line.lower():
            tail = line.lower().split("force =", 1)[1]
            values = NUMBER_RE.findall(tail)
            if len(values) >= 3:
                vectors.append([
                    _float(value) * RY_PER_BOHR_TO_EV_PER_ANGSTROM
                    for value in values[:3]
                ])
            continue
        if vectors and re.search(r"total\s+force|total\s+stress", line, re.I):
            break
    if len(vectors) != num_atoms:
        raise ValueError(
            f"Expected {num_atoms} atomic force vectors, found {len(vectors)}"
        )
    return vectors


def _parse_stress_kbar(path: Path) -> list[list[float]]:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    blocks: list[list[list[float]]] = []
    for index, line in enumerate(lines):
        if not re.search(r"total\s+stress", line, re.I) or "kbar" not in line.lower():
            continue
        block: list[list[float]] = []
        for candidate in lines[index + 1:index + 4]:
            values = NUMBER_RE.findall(candidate)
            if len(values) < 6:
                block = []
                break
            block.append([_float(value) for value in values[-3:]])
        if len(block) == 3:
            blocks.append(block)
    if not blocks:
        raise ValueError("No complete 3x3 QE stress tensor in kbar was found")
    return blocks[-1]


def _max_component_delta(first: list[list[float]], second: list[list[float]]) -> float:
    if len(first) != len(second) or any(
        len(left) != len(right) for left, right in zip(first, second)
    ):
        raise ValueError("Component arrays have incompatible shapes")
    values = [
        abs(left_value - right_value)
        for left, right in zip(first, second)
        for left_value, right_value in zip(left, right)
    ]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Component delta is empty or non-finite")
    return max(values)


def _bundled_pseudo_blockers(preflight: dict[str, Any]) -> list[str]:
    entries = preflight.get("bundled_pseudopotentials")
    if not isinstance(entries, list) or not entries:
        return ["bundled_pseudopotential_hash_inventory_missing"]
    blockers: list[str] = []
    seen_elements: set[str] = set()
    for entry in entries:
        label = str(entry.get("element") or entry.get("filename") or "unknown")
        element = str(entry.get("element") or "")
        if not element or element in seen_elements:
            blockers.append(f"bundled_pseudopotential_element_invalid:{label}")
        seen_elements.add(element)
        path = Path(str(entry.get("path") or "")).resolve()
        expected = str(entry.get("sha256") or "").lower()
        if not path.is_file():
            blockers.append(f"bundled_pseudopotential_missing:{label}")
        elif not HASH_RE.fullmatch(expected) or _sha256(path) != expected:
            blockers.append(f"bundled_pseudopotential_hash_mismatch:{label}")
    required = set(preflight.get("required_elements") or [])
    if seen_elements != required:
        blockers.append("bundled_pseudopotential_element_inventory_mismatch")
    return blockers


def _static_input_blockers(path: Path) -> list[str]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    blockers: list[str] = []
    if not re.search(r"calculation\s*=\s*['\"]scf['\"]", text, re.I):
        blockers.append("qe_input_not_static_scf")
    if re.search(r"^\s*&(?:ions|cell)\b", text, re.I | re.M):
        blockers.append("qe_input_contains_geometry_optimization_section")
    for key in ("tstress", "tprnfor"):
        if not re.search(rf"\b{key}\s*=\s*\.true\.", text, re.I):
            blockers.append(f"qe_input_missing_{key}")
    return blockers


def _verify_preflight(preflight_path: Path) -> tuple[dict[str, Any], str, str]:
    preflight_path = Path(preflight_path).resolve()
    preflight = _load_json(preflight_path)
    if preflight.get("workflow_version") != WORKFLOW_VERSION:
        raise ValueError("Not a qe_pbe_convergence_sweep_v1 preflight")
    if preflight.get("status") != "runnable_not_started":
        raise ValueError(
            f"Convergence preflight is not runnable: {preflight.get('status')}"
        )
    if preflight.get("calculation") != "scf" or preflight.get("stage") != "sweep":
        raise ValueError("Expected a static-SCF convergence sweep preflight")
    payload = preflight.get("convergence_settings_payload")
    if not isinstance(payload, dict):
        raise ValueError("Convergence settings payload is missing")
    settings_hash = str(preflight.get("convergence_settings_hash") or "").lower()
    if not HASH_RE.fullmatch(settings_hash) or convergence_settings_hash(payload) != settings_hash:
        raise ValueError("Convergence settings hash mismatch")
    if payload.get("study_points") != preflight.get("study_points"):
        raise ValueError("Preflight study points differ from the locked settings payload")
    if payload.get("representatives") != preflight.get("representatives"):
        raise ValueError("Preflight representatives differ from the locked settings payload")
    for key in ("required_elements", "covered_elements", "coverage_complete"):
        if payload.get(key) != preflight.get(key):
            raise ValueError(f"Preflight {key} differs from the locked settings payload")
    artifact_checks = (
        ("source_preflight", "source_preflight_sha256"),
        ("source_queue_manifest", "source_queue_manifest_sha256"),
        ("config", None),
        ("pseudo_manifest", None),
        ("protocol", "protocol_sha256"),
    )
    payload_hash_keys = {
        "config": "config_sha256",
        "pseudo_manifest": "pseudo_manifest_sha256",
    }
    for path_key, preflight_hash_key in artifact_checks:
        path = Path(str(preflight.get(path_key) or "")).resolve()
        if not path.is_file():
            raise ValueError(f"Locked convergence artifact is missing: {path_key}")
        expected = (
            str(preflight.get(preflight_hash_key) or "").lower()
            if preflight_hash_key else str(payload.get(payload_hash_keys[path_key]) or "").lower()
        )
        if not HASH_RE.fullmatch(expected) or _sha256(path) != expected:
            raise ValueError(f"Locked convergence artifact hash mismatch: {path_key}")
    validated_protocol = validate_protocol(_load_json(Path(preflight["protocol"])))
    if (
        validated_protocol != preflight.get("protocol_values")
        or validated_protocol != payload.get("protocol_values")
    ):
        raise ValueError("Preflight protocol values differ from the locked snapshot")
    bundled = preflight.get("bundled_pseudopotentials") or []
    bundled_inventory = [
        {
            "element": entry.get("element"),
            "filename": entry.get("filename"),
            "sha256": entry.get("sha256"),
        }
        for entry in bundled
    ]
    if bundled_inventory != payload.get("pseudopotentials"):
        raise ValueError("Bundled pseudopotentials differ from the locked inventory")
    queue_path = Path(str(preflight.get("queue_manifest") or "")).resolve()
    queue_sha = str(preflight.get("queue_manifest_sha256") or "").lower()
    if not queue_path.is_file() or not HASH_RE.fullmatch(queue_sha) or _sha256(queue_path) != queue_sha:
        raise ValueError("Convergence queue hash mismatch")
    return preflight, _sha256(preflight_path), settings_hash


def _queue_point_blockers(
    row: dict[str, str], canonical: dict[str, Any]
) -> list[str]:
    blockers: list[str] = []
    exact = (
        "point_id", "sweep_axis", "representative_id", "source_id", "formula",
        "source_cif_sha256", "copied_cif_sha256",
    )
    for key in exact:
        if str(row.get(key) or "") != str(canonical.get(key) or ""):
            blockers.append(f"queue_point_mismatch:{key}")
    for key in ("rank", "level_index", "cutoff_level_index", "kpoint_level_index"):
        try:
            if int(row.get(key, "")) != int(canonical[key]):
                blockers.append(f"queue_point_mismatch:{key}")
        except (TypeError, ValueError):
            blockers.append(f"queue_point_invalid:{key}")
    for key in (
        "num_atoms", "cutoff_pair_multiplier",
        "requested_kpoint_spacing_inv_angstrom", "ecutwfc_ry", "ecutrho_ry",
    ):
        try:
            value = float(row.get(key, ""))
            if not math.isfinite(value) or not math.isclose(
                value, float(canonical[key]), rel_tol=1e-12, abs_tol=1e-10
            ):
                blockers.append(f"queue_point_mismatch:{key}")
        except (TypeError, ValueError):
            blockers.append(f"queue_point_invalid:{key}")
    expected_grid = "x".join(str(value) for value in canonical["kpoints_grid"])
    if row.get("kpoints_grid") != expected_grid:
        blockers.append("queue_point_mismatch:kpoints_grid")
    if row.get("candidate_id") != row.get("point_id"):
        blockers.append("queue_candidate_id_not_point_id")
    if row.get("entry_id") != f"convergence:{row.get('point_id', '')}":
        blockers.append("queue_entry_id_invalid")
    if row.get("entry_role") != "convergence":
        blockers.append("queue_entry_role_not_convergence")
    if row.get("job_status") != "runnable_not_started":
        blockers.append("queue_job_status_not_runnable")
    try:
        if json.loads(row.get("blockers") or "[]") != []:
            blockers.append("queue_has_blockers")
    except json.JSONDecodeError:
        blockers.append("queue_blockers_invalid")
    return blockers


def _run_record_blockers(
    *, row: dict[str, str], qe_input: Path, qe_output: Path,
    parsed: dict[str, Any], preflight_sha256: str,
    queue_sha256: str, settings_hash: str,
) -> tuple[list[str], dict[str, Any] | None]:
    record_path = Path(str(row.get("run_record") or "")).resolve()
    if not record_path.is_file():
        return ["completed_run_record_missing"], None
    try:
        record = _load_json(record_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return ["completed_run_record_invalid"], None
    blockers: list[str] = []
    if record.get("run_status") != "completed_requires_collection":
        blockers.append("completed_run_record_status_invalid")
    for key in ("entry_id", "entry_role", "source_id", "candidate_id", "formula"):
        if record.get(key) != row.get(key):
            blockers.append(f"run_record_identity_mismatch:{key}")
    expected = {
        "qe_input_sha256": _sha256(qe_input) if qe_input.is_file() else "",
        "qe_output_sha256": _sha256(qe_output) if qe_output.is_file() else "",
        "preflight_sha256": preflight_sha256,
        "queue_manifest_sha256": queue_sha256,
        "preflight_settings_hash": settings_hash,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            blockers.append(f"run_record_mismatch:{key}")
    if record.get("qe_program_version") != parsed.get("program_version"):
        blockers.append("run_record_qe_version_mismatch")
    if not HASH_RE.fullmatch(
        str(record.get("pw_executable_sha256") or "").lower()
    ):
        blockers.append("run_record_executable_hash_missing")
    return blockers, record


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    preferred = [
        "rank", "point_id", "representative_id", "sweep_axis", "level_index",
        "cutoff_level_index", "kpoint_level_index", "formula", "num_atoms",
        "cutoff_pair_multiplier", "requested_kpoint_spacing_inv_angstrom",
        "kpoints_grid", "ecutwfc_ry", "ecutrho_ry", "total_energy_ry",
        "energy_ev_per_atom", "max_force_delta_to_anchor_ev_per_angstrom",
        "max_stress_delta_to_anchor_kbar", "energy_delta_to_anchor_mev_per_atom",
        "force_components_ev_per_angstrom", "stress_components_kbar",
        "point_within_tolerances", "stable_tail_from_this_level",
        "convergence_gate_status", "gate_failures", "qe_program_version",
        "pw_executable_sha256", "qe_input_sha256", "qe_output_sha256",
    ]
    extras = sorted({key for row in rows for key in row} - set(preferred))
    fields = preferred + extras
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def _select_stable_tail(
    rows: list[dict[str, Any]], *, min_tail: int,
) -> dict[str, Any] | None:
    for index, row in enumerate(rows):
        tail = rows[index:]
        if len(tail) >= min_tail and all(
            item.get("point_within_tolerances") is True for item in tail
        ):
            return row
    return None


def collect_convergence(
    *, preflight_path: Path, output_dir: Path,
) -> dict[str, Any]:
    preflight_path = Path(preflight_path).resolve()
    preflight, preflight_sha, settings_hash = _verify_preflight(preflight_path)
    queue_path = Path(preflight["queue_manifest"]).resolve()
    queue_sha = str(preflight["queue_manifest_sha256"])
    queue = _read_csv(queue_path)
    common_failures = _bundled_pseudo_blockers(preflight)
    canonical_points = {
        point["point_id"]: point for point in preflight["study_points"]
    }
    if len(canonical_points) != len(preflight["study_points"]):
        raise ValueError("Duplicate point IDs in convergence preflight")
    queue_ids = [row.get("point_id", "") for row in queue]
    if set(queue_ids) != set(canonical_points) or len(queue_ids) != len(set(queue_ids)):
        raise ValueError("Queue point IDs differ from the locked convergence study")

    rows: list[dict[str, Any]] = []
    force_by_point: dict[str, list[list[float]]] = {}
    stress_by_point: dict[str, list[list[float]]] = {}
    for queue_row in sorted(queue, key=lambda item: int(item["rank"])):
        point_id = queue_row["point_id"]
        canonical = canonical_points[point_id]
        failures = list(common_failures)
        failures.extend(_queue_point_blockers(queue_row, canonical))
        if queue_row.get("convergence_settings_hash") != settings_hash:
            failures.append("queue_convergence_settings_hash_mismatch")
        qe_input = Path(str(queue_row.get("qe_input") or "")).resolve()
        qe_output = Path(str(queue_row.get("qe_output") or "")).resolve()
        copied_cif = Path(str(queue_row.get("copied_cif") or "")).resolve()
        for path, expected, label in (
            (qe_input, queue_row.get("qe_input_sha256", ""), "qe_input"),
            (copied_cif, queue_row.get("copied_cif_sha256", ""), "copied_cif"),
        ):
            expected = str(expected).lower()
            if not path.is_file():
                failures.append(f"{label}_missing")
            elif not HASH_RE.fullmatch(expected) or _sha256(path) != expected:
                failures.append(f"{label}_hash_mismatch")
        failures.extend(_static_input_blockers(qe_input))
        parsed = summarize_qe_output(qe_output)
        run_failures, run_record = _run_record_blockers(
            row=queue_row,
            qe_input=qe_input,
            qe_output=qe_output,
            parsed=parsed,
            preflight_sha256=preflight_sha,
            queue_sha256=queue_sha,
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
        total_energy_ry = parsed.get("total_energy_ry")
        if total_energy_ry is None or not math.isfinite(float(total_energy_ry)):
            failures.append("finite_total_energy_missing")
        canonical_num_atoms = float(canonical["num_atoms"])
        num_atoms = int(round(canonical_num_atoms))
        if (
            not math.isfinite(canonical_num_atoms)
            or num_atoms < 1
            or not math.isclose(canonical_num_atoms, num_atoms, abs_tol=1e-9)
        ):
            failures.append("canonical_atom_count_invalid")
            num_atoms = max(num_atoms, 1)
        try:
            force_by_point[point_id] = _parse_force_components(qe_output, num_atoms)
        except (OSError, ValueError) as exc:
            failures.append(f"force_components_invalid:{exc}")
        try:
            stress_by_point[point_id] = _parse_stress_kbar(qe_output)
        except (OSError, ValueError) as exc:
            failures.append(f"stress_components_invalid:{exc}")
        program_version = str(parsed.get("program_version") or "").strip()
        if not program_version:
            failures.append("qe_program_version_missing")
        executable_sha = str(
            (run_record or {}).get("pw_executable_sha256") or ""
        ).lower()
        energy_ev_per_atom: float | str = ""
        if total_energy_ry is not None and math.isfinite(float(total_energy_ry)):
            energy_ev_per_atom = float(total_energy_ry) * RY_TO_EV / num_atoms
        rows.append({
            **canonical,
            "kpoints_grid": "x".join(str(value) for value in canonical["kpoints_grid"]),
            "total_energy_ry": total_energy_ry if total_energy_ry is not None else "",
            "energy_ev_per_atom": energy_ev_per_atom,
            "force_components_ev_per_angstrom": (
                json.dumps(force_by_point[point_id])
                if point_id in force_by_point else ""
            ),
            "stress_components_kbar": (
                json.dumps(stress_by_point[point_id])
                if point_id in stress_by_point else ""
            ),
            "max_force_delta_to_anchor_ev_per_angstrom": "",
            "max_stress_delta_to_anchor_kbar": "",
            "energy_delta_to_anchor_mev_per_atom": "",
            "point_within_tolerances": False,
            "stable_tail_from_this_level": False,
            "convergence_gate_status": (
                "converged_output" if not failures else "blocked"
            ),
            "gate_failures": json.dumps(sorted(set(failures))),
            "qe_program_version": program_version,
            "pw_executable_sha256": executable_sha,
            "qe_input_sha256": _sha256(qe_input) if qe_input.is_file() else "",
            "qe_output_sha256": _sha256(qe_output) if qe_output.is_file() else "",
        })

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    blocked = [row for row in rows if row["convergence_gate_status"] != "converged_output"]
    versions = {row["qe_program_version"] for row in rows if row["qe_program_version"]}
    executables = {
        row["pw_executable_sha256"]
        for row in rows if HASH_RE.fullmatch(row["pw_executable_sha256"])
    }
    global_failures: list[str] = []
    if len(versions) != 1:
        global_failures.append("mixed_or_missing_qe_versions")
    if len(executables) != 1:
        global_failures.append("mixed_or_missing_qe_executables")
    protocol = preflight["protocol_values"]
    energy_tol = float(protocol["energy_tolerance_mev_per_atom"])
    force_tol = float(protocol["force_component_tolerance_ev_per_angstrom"])
    stress_tol = float(protocol["stress_component_tolerance_kbar"])
    min_tail = int(protocol["min_stable_tail_points"])
    selections_by_representative: dict[str, Any] = {}
    needs_cutoff = False
    needs_kpoint = False
    if not blocked and not global_failures:
        by_rep: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_rep[row["representative_id"]].append(row)
        for representative_id, rep_rows in sorted(by_rep.items()):
            anchors = [row for row in rep_rows if row["sweep_axis"] == "anchor"]
            if len(anchors) != 1:
                global_failures.append(f"anchor_count_invalid:{representative_id}")
                continue
            anchor = anchors[0]
            anchor_id = anchor["point_id"]
            for row in rep_rows:
                point_id = row["point_id"]
                energy_delta = abs(
                    float(row["energy_ev_per_atom"])
                    - float(anchor["energy_ev_per_atom"])
                ) * 1000.0
                force_delta = _max_component_delta(
                    force_by_point[point_id], force_by_point[anchor_id]
                )
                stress_delta = _max_component_delta(
                    stress_by_point[point_id], stress_by_point[anchor_id]
                )
                within = (
                    energy_delta <= energy_tol
                    and force_delta <= force_tol
                    and stress_delta <= stress_tol
                )
                row.update({
                    "energy_delta_to_anchor_mev_per_atom": energy_delta,
                    "max_force_delta_to_anchor_ev_per_angstrom": force_delta,
                    "max_stress_delta_to_anchor_kbar": stress_delta,
                    "point_within_tolerances": within,
                })
            cutoff_rows = sorted(
                [row for row in rep_rows if row["sweep_axis"] in {"cutoff_pair", "anchor"}],
                key=lambda row: int(row["cutoff_level_index"]),
            )
            kpoint_rows = sorted(
                [row for row in rep_rows if row["sweep_axis"] in {"kpoint", "anchor"}],
                key=lambda row: int(row["kpoint_level_index"]),
            )
            selected_cutoff = _select_stable_tail(cutoff_rows, min_tail=min_tail)
            selected_kpoint = _select_stable_tail(kpoint_rows, min_tail=min_tail)
            if selected_cutoff is None:
                needs_cutoff = True
            else:
                start = cutoff_rows.index(selected_cutoff)
                for row in cutoff_rows[start:]:
                    row["stable_tail_from_this_level"] = True
            if selected_kpoint is None:
                needs_kpoint = True
            else:
                start = kpoint_rows.index(selected_kpoint)
                for row in kpoint_rows[start:]:
                    row["stable_tail_from_this_level"] = True
            selections_by_representative[representative_id] = {
                "selected_cutoff_pair": (
                    {
                        "multiplier": selected_cutoff["cutoff_pair_multiplier"],
                        "ecutwfc_ry": selected_cutoff["ecutwfc_ry"],
                        "ecutrho_ry": selected_cutoff["ecutrho_ry"],
                        "point_id": selected_cutoff["point_id"],
                    } if selected_cutoff else None
                ),
                "selected_kpoint": (
                    {
                        "spacing_inv_angstrom": selected_kpoint[
                            "requested_kpoint_spacing_inv_angstrom"
                        ],
                        "kpoints_grid": selected_kpoint["kpoints_grid"],
                        "point_id": selected_kpoint["point_id"],
                    } if selected_kpoint else None
                ),
                "anchor_point_id": anchor_id,
            }

    if blocked or global_failures:
        status = "blocked_incomplete_or_failed_runs"
    elif needs_cutoff:
        status = "needs_higher_cutoff"
    elif needs_kpoint:
        status = "needs_denser_kpoint_grid"
    else:
        status = "provisional_selection_ready"
    global_selection: dict[str, Any] | None = None
    if status == "provisional_selection_ready":
        cutoff_values = [
            value["selected_cutoff_pair"]
            for value in selections_by_representative.values()
        ]
        kpoint_values = [
            value["selected_kpoint"]
            for value in selections_by_representative.values()
        ]
        global_selection = {
            "ecutwfc_ry": max(value["ecutwfc_ry"] for value in cutoff_values),
            "ecutrho_ry": max(value["ecutrho_ry"] for value in cutoff_values),
            "cutoff_pair_multiplier": max(
                value["multiplier"] for value in cutoff_values
            ),
            "kpoint_spacing_inv_angstrom": min(
                value["spacing_inv_angstrom"] for value in kpoint_values
            ),
        }

    results_path = output_dir / "convergence_results.csv"
    _write_csv(results_path, rows)
    summary = {
        "collector_version": COLLECTOR_VERSION,
        "status": status,
        "candidate_count": len(preflight.get("representatives") or []),
        "point_count": len(rows),
        "gate_status_counts": dict(Counter(row["convergence_gate_status"] for row in rows)),
        "global_failures": sorted(set(global_failures)),
        "protocol": protocol,
        "convergence_settings_hash": settings_hash,
        "qe_program_version": next(iter(versions)) if len(versions) == 1 else None,
        "pw_executable_sha256": next(iter(executables)) if len(executables) == 1 else None,
        "coverage_complete": preflight.get("coverage_complete") is True,
        "cutoff_window_converged": (
            None if blocked or global_failures else not needs_cutoff
        ),
        "kpoint_window_converged": (
            None if blocked or global_failures else not needs_kpoint
        ),
        "confirmation_eligible": (
            preflight.get("confirmation_eligible") is True
            and status == "provisional_selection_ready"
        ),
        "certificate_eligible": False,
        "selections_by_representative": selections_by_representative,
        "provisional_global_selection": global_selection,
        "convergence_preflight": str(preflight_path),
        "convergence_preflight_sha256": preflight_sha,
        "candidate_source_preflight": preflight.get("source_preflight"),
        "candidate_source_preflight_sha256": preflight.get("source_preflight_sha256"),
        "results": str(results_path),
        "results_sha256": _sha256(results_path),
        "confirmation_required": True,
        "scientific_limit": (
            "This is a provisional selection over the tested sparse window. "
            "It must not alter production settings until an independent "
            "selected-combination confirmation passes."
        ),
        "points": rows,
    }
    summary_path = output_dir / "convergence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Convergence status: {status}")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    collect_convergence(preflight_path=args.preflight, output_dir=args.output_dir)
