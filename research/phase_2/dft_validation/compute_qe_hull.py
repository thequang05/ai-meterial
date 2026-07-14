"""Compute audited finite-smearing PBE hull screens from consistent QE energies.

Coverage-complete energy-above-hull values are emitted only when a manifest
proves that every non-empty subsystem was queried and every declared reference
entry was recomputed successfully with the same static settings.  Incomplete
inventories are blocked; ``--diagnostic-incomplete`` may emit a clearly named
lower-bound diagnostic, never a coverage-complete hull result or pass label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pymatgen.core import Composition


WORKFLOW_VERSION = "qe_reference_hull_v1"
CLAIM_SCOPE = (
    "finite-smearing, non-spin-polarized, scalar-relativistic PBE screening "
    "hull relative to the declared, recomputed reference snapshot"
)
HASH_RE = re.compile(r"[0-9a-f]{64}")
RY_TO_EV = 13.605693122994
REFERENCE_INVENTORY_VERSION = "mp2019_reference_inventory_v1"
MAX_PROTOCOL_ELEMENTS = 6


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


def _verify_energy_summary(entries_path: Path, summary_path: Path) -> dict[str, Any]:
    summary = _load_json(summary_path)
    if summary.get("collector_version") != "qe_static_collector_v1":
        raise ValueError("Energy summary was not produced by qe_static_collector_v1")
    expected_sha = str(summary.get("results_manifest_sha256") or "").lower()
    if not HASH_RE.fullmatch(expected_sha) or _sha256(entries_path) != expected_sha:
        raise ValueError("Static energy CSV no longer matches its collector summary")
    records_digest = str(summary.get("energy_records_sha256") or "").lower()
    if not HASH_RE.fullmatch(records_digest):
        raise ValueError("Energy summary has no valid canonical records digest")
    return summary


def _receipt(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_snapshot_manifest_payload(snapshot: dict[str, Any]) -> None:
    """Require the complete attestation emitted by the snapshot verifier."""
    if (
        snapshot.get("schema_version") != "mp_snapshot_manifest_v1"
        or snapshot.get("status") != "verified_exact_id_sets"
        or snapshot.get("snapshot_id") != "mp.2019.04.01"
        or snapshot.get("composition_audit_status")
        != "all_cache_cifs_match_metadata"
    ):
        raise ValueError("Invalid locked MP snapshot manifest identity/status")
    material_count = snapshot.get("material_count")
    if (
        isinstance(material_count, bool)
        or not isinstance(material_count, int)
        or material_count < 1
    ):
        raise ValueError("Snapshot manifest has no positive integer material_count")
    material_id_sha = str(snapshot.get("material_id_sha256") or "").lower()
    composition_audit_sha = str(
        snapshot.get("composition_audit_sha256") or ""
    ).lower()
    if not HASH_RE.fullmatch(material_id_sha):
        raise ValueError("Snapshot manifest material-ID digest is invalid")
    if not HASH_RE.fullmatch(composition_audit_sha):
        raise ValueError("Snapshot manifest composition-audit digest is invalid")

    for section_name in ("raw_json", "materials_csv", "structure_cache"):
        section = snapshot.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"Snapshot manifest section is missing: {section_name}")
        if not str(section.get("path") or "").strip():
            raise ValueError(f"Snapshot manifest path is missing: {section_name}")
        if not HASH_RE.fullmatch(str(section.get("sha256") or "").lower()):
            raise ValueError(f"Snapshot manifest file digest is invalid: {section_name}")
        if section.get("material_count") != material_count:
            raise ValueError(f"Snapshot manifest count mismatch: {section_name}")
        if str(section.get("material_id_sha256") or "").lower() != material_id_sha:
            raise ValueError(f"Snapshot manifest ID digest mismatch: {section_name}")
    if snapshot["raw_json"].get("material_id_occurrences") != material_count:
        raise ValueError("Snapshot raw material-ID occurrence count mismatch")
    if (
        str(
            snapshot["structure_cache"].get("composition_audit_sha256") or ""
        ).lower()
        != composition_audit_sha
    ):
        raise ValueError("Snapshot composition-audit digest mismatch")


def _verify_reference_inventory(
    *,
    inventory_path: Path,
    coverage: dict[str, Any],
    references: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    inventory_path = Path(inventory_path).resolve()
    inventory_meta = coverage.get("inventory")
    if not isinstance(inventory_meta, dict):
        raise ValueError("Coverage inventory metadata is missing")
    expected_file_sha = str(
        inventory_meta.get("reference_inventory_sha256") or ""
    ).lower()
    if not HASH_RE.fullmatch(expected_file_sha) or _sha256(inventory_path) != expected_file_sha:
        raise ValueError("Reference inventory CSV does not match the coverage manifest")
    snapshot_path = Path(str(inventory_meta.get("snapshot_manifest") or "")).resolve()
    snapshot_sha = str(inventory_meta.get("snapshot_manifest_sha256") or "").lower()
    if (
        not snapshot_path.is_file()
        or not HASH_RE.fullmatch(snapshot_sha)
        or _sha256(snapshot_path) != snapshot_sha
    ):
        raise ValueError("Locked MP snapshot manifest is missing or modified")
    snapshot = _load_json(snapshot_path)
    _validate_snapshot_manifest_payload(snapshot)
    if (
        str(inventory_meta.get("materials_csv_sha256") or "").lower()
        != str(snapshot["materials_csv"]["sha256"]).lower()
        or str(inventory_meta.get("structure_cache_sha256") or "").lower()
        != str(snapshot["structure_cache"]["sha256"]).lower()
    ):
        raise ValueError("Coverage source hashes do not match the locked snapshot")

    rows = _read_csv(inventory_path)
    by_id: dict[str, dict[str, str]] = {}
    selected: list[dict[str, str]] = []
    for row in rows:
        entry_id = str(row.get("entry_id") or "").strip()
        source_id = str(row.get("source_id") or "").strip()
        formula = str(row.get("formula") or "").strip()
        subsystem = str(row.get("subsystem") or "").strip()
        source_cif_sha = str(row.get("source_cif_sha256") or "").lower()
        if not entry_id or entry_id in by_id:
            raise ValueError(f"Invalid/duplicate inventory entry_id: {entry_id!r}")
        if row.get("entry_role") != "reference":
            raise ValueError(f"Inventory entry is not a reference: {entry_id}")
        if str(row.get("cif_materialized") or "").lower() not in {"true", "1", "yes"}:
            raise ValueError(f"Inventory CIF is not materialized: {entry_id}")
        if not source_id or not formula or not HASH_RE.fullmatch(source_cif_sha):
            raise ValueError(f"Inventory lineage fields are invalid: {entry_id}")
        if _chemsys({element.symbol for element in Composition(formula).elements}) != subsystem:
            raise ValueError(f"Inventory subsystem mismatch: {entry_id}")
        by_id[entry_id] = row
        selected.append({
            "entry_id": entry_id,
            "source_id": source_id,
            "formula": formula,
            "subsystem": subsystem,
            "source_cif_sha256": source_cif_sha,
        })
    computed_inventory_digest = _receipt({
        "inventory_version": REFERENCE_INVENTORY_VERSION,
        "materials_csv_sha256": inventory_meta.get("materials_csv_sha256"),
        "structure_cache_sha256": inventory_meta.get("structure_cache_sha256"),
        "selected": sorted(selected, key=lambda row: row["entry_id"]),
        "diagnostic_max_per_subsystem": inventory_meta.get(
            "diagnostic_max_per_subsystem"
        ),
    })
    if computed_inventory_digest != inventory_meta.get("inventory_sha256"):
        raise ValueError("Canonical reference inventory digest mismatch")
    for reference in references:
        expected = by_id.get(reference["entry_id"])
        if expected is None:
            raise ValueError(
                f"Static reference is absent from locked inventory: {reference['entry_id']}"
            )
        if (
            reference["source_id"] != expected["source_id"]
            or reference["formula"] != Composition(expected["formula"]).reduced_formula
            or reference["source_cif_sha256"] != expected["source_cif_sha256"].lower()
        ):
            raise ValueError(
                f"Static reference lineage mismatch: {reference['entry_id']}"
            )
    return by_id


def _chemsys(elements: set[str] | frozenset[str]) -> str:
    return "-".join(sorted(elements))


def _subsystem_keys(elements: set[str]) -> set[str]:
    if not elements:
        raise ValueError("Chemical system must contain at least one element")
    if len(elements) > MAX_PROTOCOL_ELEMENTS:
        raise ValueError(
            f"Protocol v1 supports at most {MAX_PROTOCOL_ELEMENTS} elements per "
            f"candidate; received {len(elements)} to avoid a 2^N subsystem explosion"
        )
    ordered = sorted(elements)
    return {
        "-".join(combo)
        for size in range(1, len(ordered) + 1)
        for combo in itertools.combinations(ordered, size)
    }


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value for {label}: {value!r}")
    return result


def _parse_entry(row: dict[str, str], expected_role: str) -> dict[str, Any]:
    entry_id = str(row.get("entry_id") or "").strip()
    if not entry_id:
        raise ValueError("Energy row has no entry_id")
    role = str(row.get("entry_role") or "").strip()
    if role != expected_role:
        raise ValueError(
            f"Entry {entry_id} has role={role!r}; expected {expected_role!r}"
        )
    formula = str(row.get("formula") or "").strip()
    if not formula:
        raise ValueError(f"Entry {entry_id} has no formula")
    try:
        raw_composition = json.loads(row.get("composition_json") or "")
        if not isinstance(raw_composition, dict) or not raw_composition:
            raise ValueError("composition_json must be a non-empty object")
        composition = Composition({
            str(symbol): _finite_float(amount, f"{entry_id}.composition.{symbol}")
            for symbol, amount in raw_composition.items()
        })
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid composition_json for {entry_id}: {exc}") from exc
    if any(amount <= 0 for amount in composition.values()):
        raise ValueError(f"Entry {entry_id} has a non-positive composition amount")
    if any(abs(float(amount) - round(float(amount))) > 1e-8 for amount in composition.values()):
        raise ValueError(f"Entry {entry_id} has fractional cell atom counts")
    if composition.reduced_formula != Composition(formula).reduced_formula:
        raise ValueError(
            f"Formula/composition mismatch for {entry_id}: "
            f"formula={formula}, composition={composition.formula}"
        )
    num_atoms = _finite_float(row.get("num_atoms"), f"{entry_id}.num_atoms")
    if (
        num_atoms <= 0
        or abs(num_atoms - round(num_atoms)) > 1e-8
        or abs(num_atoms - composition.num_atoms) > 1e-6
    ):
        raise ValueError(f"Atom count mismatch for {entry_id}")

    status = str(row.get("static_gate_status") or "").strip()
    converged = status == "dft_static_converged"
    total_energy_ev: float | None = None
    energy_ev_per_atom: float | None = None
    total_energy_ry: float | None = None
    settings_hash = str(row.get("static_settings_hash") or "").strip().lower()
    if converged:
        total_energy_ry = _finite_float(
            row.get("total_energy_ry"), f"{entry_id}.total_energy_ry"
        )
        total_energy_ev = _finite_float(
            row.get("total_energy_ev"), f"{entry_id}.total_energy_ev"
        )
        energy_ev_per_atom = _finite_float(
            row.get("energy_ev_per_atom"), f"{entry_id}.energy_ev_per_atom"
        )
        energy_conversion_tolerance = max(1e-8, abs(total_energy_ev) * 1e-10)
        if abs(total_energy_ry * RY_TO_EV - total_energy_ev) > energy_conversion_tolerance:
            raise ValueError(f"Ry/eV energy mismatch for {entry_id}")
        tolerance = max(1e-8, abs(energy_ev_per_atom) * 1e-10)
        if abs(total_energy_ev / num_atoms - energy_ev_per_atom) > tolerance:
            raise ValueError(f"Per-atom energy mismatch for {entry_id}")
        if not HASH_RE.fullmatch(settings_hash):
            raise ValueError(f"Invalid static_settings_hash for {entry_id}")
        source_cif_sha256 = str(row.get("source_cif_sha256") or "").lower()
        relaxed_cif_sha256 = str(row.get("relaxed_cif_sha256") or "").lower()
        qe_input_sha256 = str(row.get("qe_input_sha256") or "").lower()
        qe_output_sha256 = str(row.get("qe_output_sha256") or "").lower()
        pw_executable_sha256 = str(row.get("pw_executable_sha256") or "").lower()
        for label, digest in (
            ("source_cif_sha256", source_cif_sha256),
            ("relaxed_cif_sha256", relaxed_cif_sha256),
            ("qe_input_sha256", qe_input_sha256),
            ("qe_output_sha256", qe_output_sha256),
            ("pw_executable_sha256", pw_executable_sha256),
        ):
            if not HASH_RE.fullmatch(digest):
                raise ValueError(f"Invalid {label} for {entry_id}")
        canonical_composition = {
            element.symbol: float(amount)
            for element, amount in composition.items()
        }
        expected_lineage = hashlib.sha256(
            json.dumps({
                "entry_id": entry_id,
                "entry_role": role,
                "source_id": str(row.get("source_id") or "").strip(),
                "formula": composition.reduced_formula,
                "composition_json": json.dumps(
                    canonical_composition, sort_keys=True
                ),
                "num_atoms": num_atoms,
                "source_cif_sha256": source_cif_sha256,
                "relaxed_cif_sha256": relaxed_cif_sha256,
                "qe_input_sha256": qe_input_sha256,
                "qe_output_sha256": qe_output_sha256,
                "static_settings_hash": settings_hash,
                "pw_executable_sha256": pw_executable_sha256,
                "total_energy_ry": total_energy_ry,
            }, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if str(row.get("entry_lineage_sha256") or "").lower() != expected_lineage:
            raise ValueError(f"Entry lineage hash mismatch for {entry_id}")
    return {
        "entry_id": entry_id,
        "entry_role": role,
        "source_id": str(row.get("source_id") or "").strip(),
        "candidate_id": str(row.get("candidate_id") or "").strip(),
        "formula": composition.reduced_formula,
        "composition": composition,
        "elements": frozenset(element.symbol for element in composition.elements),
        "num_atoms": num_atoms,
        "total_energy_ev": total_energy_ev,
        "total_energy_ry": total_energy_ry,
        "energy_ev_per_atom": energy_ev_per_atom,
        "static_gate_status": status,
        "converged": converged,
        "settings_hash": settings_hash,
        "source_cif_sha256": str(row.get("source_cif_sha256") or "").lower(),
        "relaxed_cif_sha256": str(row.get("relaxed_cif_sha256") or "").lower(),
        "entry_lineage_sha256": str(row.get("entry_lineage_sha256") or "").lower(),
    }


def _coverage_gate(
    *,
    candidate: dict[str, Any],
    references: list[dict[str, Any]],
    coverage: dict[str, Any],
    reference_inventory: dict[str, dict[str, str]],
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    blockers: list[str] = []
    elements = set(candidate["elements"])
    system_key = _chemsys(elements)
    systems = coverage.get("systems")
    if not isinstance(systems, dict) or not isinstance(systems.get(system_key), dict):
        return [f"coverage_system_missing:{system_key}"], [], {
            "chemsys": system_key,
            "expected_reference_ids": [],
            "computed_reference_ids": [],
        }
    system = systems[system_key]
    declared_elements = set(system.get("elements") or [])
    if declared_elements != elements:
        blockers.append("coverage_element_set_mismatch")
    if system.get("coverage_complete_assertion") is not True:
        blockers.append("coverage_complete_assertion_missing")

    subsystems = system.get("subsystems")
    required_subsystems = _subsystem_keys(elements)
    if not isinstance(subsystems, dict):
        subsystems = {}
        blockers.append("subsystem_inventory_missing")
    actual_subsystem_keys = set(subsystems)
    for missing in sorted(required_subsystems - actual_subsystem_keys):
        blockers.append(f"subsystem_missing:{missing}")
    for extra in sorted(actual_subsystem_keys - required_subsystems):
        blockers.append(f"unexpected_subsystem:{extra}")

    expected_ids: set[str] = set()
    expected_to_subsystem: dict[str, str] = {}
    for subsystem_key in sorted(required_subsystems):
        record = subsystems.get(subsystem_key)
        if not isinstance(record, dict):
            continue
        if record.get("query_status") != "complete":
            blockers.append(f"subsystem_query_incomplete:{subsystem_key}")
        receipt = str(record.get("query_receipt_sha256") or "").lower()
        if not HASH_RE.fullmatch(receipt):
            blockers.append(f"subsystem_query_receipt_invalid:{subsystem_key}")
        ids = record.get("expected_reference_ids")
        if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
            blockers.append(f"subsystem_reference_ids_invalid:{subsystem_key}")
            continue
        if len(subsystem_key.split("-")) == 1 and not ids:
            blockers.append(f"elemental_reference_missing:{subsystem_key}")
        inventory_ids = {
            entry_id for entry_id, inventory_row in reference_inventory.items()
            if inventory_row.get("subsystem") == subsystem_key
        }
        if set(ids) != inventory_ids:
            blockers.append(f"subsystem_inventory_id_mismatch:{subsystem_key}")
        expected_receipt = _receipt({
            "inventory_sha256": coverage.get("inventory", {}).get("inventory_sha256"),
            "subsystem": subsystem_key,
            "expected_reference_ids": ids,
            "query_status": record.get("query_status"),
        })
        if receipt != expected_receipt:
            blockers.append(f"subsystem_query_receipt_mismatch:{subsystem_key}")
        for entry_id in ids:
            if entry_id in expected_to_subsystem:
                blockers.append(f"duplicate_expected_reference_id:{entry_id}")
            expected_to_subsystem[entry_id] = subsystem_key
            expected_ids.add(entry_id)

    deduplications = system.get("deduplications") or []
    if not isinstance(deduplications, list):
        blockers.append("deduplications_invalid")
        deduplications = []
    if deduplications:
        blockers.append("deduplication_not_supported_by_coverage_v1")
    excluded_ids: set[str] = set()
    effective_expected_ids = expected_ids

    relevant_all = [ref for ref in references if ref["elements"].issubset(elements)]
    relevant_ids = {ref["entry_id"] for ref in relevant_all}
    for missing in sorted(effective_expected_ids - relevant_ids):
        blockers.append(f"reference_entry_missing:{missing}")
    for extra in sorted(relevant_ids - effective_expected_ids):
        blockers.append(f"undeclared_reference_entry:{extra}")

    by_id = {ref["entry_id"]: ref for ref in relevant_all}
    usable: list[dict[str, Any]] = []
    for entry_id in sorted(effective_expected_ids):
        ref = by_id.get(entry_id)
        if ref is None:
            continue
        expected_subsystem = expected_to_subsystem[entry_id]
        if _chemsys(set(ref["elements"])) != expected_subsystem:
            blockers.append(f"reference_subsystem_mismatch:{entry_id}")
        if not ref["converged"]:
            blockers.append(f"reference_not_converged:{entry_id}")
        elif ref["settings_hash"] != candidate["settings_hash"]:
            blockers.append(f"reference_settings_hash_mismatch:{entry_id}")
        else:
            usable.append(ref)

    manifest_hash = str(coverage.get("static_settings_hash") or "").lower()
    if manifest_hash != candidate["settings_hash"]:
        blockers.append("coverage_manifest_settings_hash_mismatch")
    audit = {
        "chemsys": system_key,
        "required_subsystems": sorted(required_subsystems),
        "expected_reference_ids": sorted(effective_expected_ids),
        "computed_reference_ids": sorted(relevant_ids),
        "excluded_exact_duplicate_ids": sorted(excluded_ids),
    }
    return blockers, usable, audit


def _inventory_blockers(coverage: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if coverage.get("schema_version") != "dft_reference_coverage_v1":
        blockers.append("coverage_schema_mismatch")
    inventory = coverage.get("inventory")
    if not isinstance(inventory, dict):
        return blockers + ["coverage_inventory_missing"]
    for field in ("source_name", "source_snapshot_id", "query_code_version", "scope"):
        if not str(inventory.get(field) or "").strip():
            blockers.append(f"coverage_inventory_field_missing:{field}")
    if not HASH_RE.fullmatch(str(inventory.get("inventory_sha256") or "").lower()):
        blockers.append("coverage_inventory_sha256_invalid")
    if inventory.get("source_snapshot_id") != "mp.2019.04.01":
        blockers.append("coverage_snapshot_id_mismatch")
    if inventory.get("query_code_version") != REFERENCE_INVENTORY_VERSION:
        blockers.append("coverage_query_code_version_mismatch")
    if inventory.get("scope") != "all structures from every non-empty subsystem":
        blockers.append("coverage_scope_policy_mismatch")
    return blockers


def _pd_entry(entry: dict[str, Any]) -> PDEntry:
    from pymatgen.analysis.phase_diagram import PDEntry

    assert entry["total_energy_ev"] is not None
    return PDEntry(
        entry["composition"],
        float(entry["total_energy_ev"]),
        name=entry["entry_id"],
    )


def _diagnostic_lower_bound(
    candidate: dict[str, Any], references: list[dict[str, Any]]
) -> float | None:
    usable = [
        ref for ref in references
        if ref["converged"]
        and ref["settings_hash"] == candidate["settings_hash"]
        and ref["elements"].issubset(candidate["elements"])
    ]
    unary = {
        next(iter(ref["elements"]))
        for ref in usable
        if len(ref["elements"]) == 1
    }
    if unary != set(candidate["elements"]):
        return None
    try:
        from pymatgen.analysis.phase_diagram import PhaseDiagram

        diagram = PhaseDiagram([_pd_entry(ref) for ref in usable])
        hull_energy = float(
            diagram.get_hull_energy_per_atom(candidate["composition"])
        )
        assert candidate["energy_ev_per_atom"] is not None
        lower_bound = float(candidate["energy_ev_per_atom"]) - hull_energy
        if not math.isfinite(hull_energy) or not math.isfinite(lower_bound):
            return None
        return max(0.0, lower_bound)
    except Exception:  # noqa: BLE001
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    preferred = [
        "candidate_id", "entry_id", "formula", "chemsys", "coverage_status",
        "coverage_blockers", "reference_set_id", "expected_reference_count",
        "computed_reference_count", "missing_reference_ids",
        "distance_to_reference_hull_ev_per_atom",
        "energy_above_augmented_hull_ev_per_atom",
        "diagnostic_lower_bound_ehull_ev_per_atom", "decomposition_json",
        "hull_class", "within_user_supplied_hull_threshold", "claim_scope",
        "static_settings_hash",
    ]
    extras = sorted({key for row in rows for key in row} - set(preferred))
    fields = preferred + extras
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def compute_hulls(
    *,
    candidate_entries_path: Path,
    candidate_summary_path: Path,
    reference_entries_path: Path,
    reference_summary_path: Path,
    reference_inventory_path: Path,
    coverage_manifest_path: Path,
    output_dir: Path,
    near_hull_threshold_ev_per_atom: float = 0.025,
    diagnostic_incomplete: bool = False,
) -> dict[str, Any]:
    if (
        not math.isfinite(near_hull_threshold_ev_per_atom)
        or near_hull_threshold_ev_per_atom < 0
    ):
        raise ValueError("near-hull threshold must be non-negative")
    candidate_entries_path = Path(candidate_entries_path).resolve()
    reference_entries_path = Path(reference_entries_path).resolve()
    candidate_summary = _verify_energy_summary(
        candidate_entries_path, candidate_summary_path
    )
    reference_summary = _verify_energy_summary(
        reference_entries_path, reference_summary_path
    )
    candidates = [
        _parse_entry(row, "candidate") for row in _read_csv(candidate_entries_path)
    ]
    references = [
        _parse_entry(row, "reference") for row in _read_csv(reference_entries_path)
    ]
    if not candidates:
        raise ValueError("Candidate entries are empty")
    candidate_hashes = {entry["settings_hash"] for entry in candidates if entry["converged"]}
    reference_hashes = {entry["settings_hash"] for entry in references if entry["converged"]}
    if candidate_summary.get("static_settings_hash") not in candidate_hashes:
        raise ValueError("Candidate summary settings hash does not match its entries")
    if reference_summary.get("static_settings_hash") not in reference_hashes:
        raise ValueError("Reference summary settings hash does not match its entries")
    all_ids = [entry["entry_id"] for entry in candidates + references]
    id_counts = Counter(all_ids)
    duplicates = sorted(entry_id for entry_id, count in id_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate entry IDs: {duplicates}")
    coverage = _load_json(coverage_manifest_path)
    reference_inventory = _verify_reference_inventory(
        inventory_path=reference_inventory_path,
        coverage=coverage,
        references=references,
    )
    global_blockers = _inventory_blockers(coverage)

    converged_candidates = [candidate for candidate in candidates if candidate["converged"]]
    result_rows: list[dict[str, Any]] = []
    audit_by_system: dict[str, Any] = {}
    for candidate in candidates:
        blockers = list(global_blockers)
        if not candidate["converged"]:
            blockers.append("candidate_static_energy_not_converged")
            usable_references: list[dict[str, Any]] = []
            audit = {
                "chemsys": _chemsys(set(candidate["elements"])),
                "expected_reference_ids": [],
                "computed_reference_ids": [],
            }
        else:
            coverage_blockers, usable_references, audit = _coverage_gate(
                candidate=candidate,
                references=references,
                coverage=coverage,
                reference_inventory=reference_inventory,
            )
            blockers.extend(coverage_blockers)
        system_key = audit["chemsys"]
        audit_by_system.setdefault(system_key, audit)
        expected_ids = set(audit.get("expected_reference_ids", []))
        computed_ids = set(audit.get("computed_reference_ids", []))
        row: dict[str, Any] = {
            "candidate_id": candidate["candidate_id"] or candidate["source_id"],
            "entry_id": candidate["entry_id"],
            "formula": candidate["formula"],
            "chemsys": system_key,
            "coverage_status": "complete" if not blockers else "blocked_incomplete",
            "coverage_blockers": json.dumps(sorted(set(blockers))),
            "reference_set_id": coverage.get("coverage_id", ""),
            "expected_reference_count": len(expected_ids),
            "computed_reference_count": len(computed_ids),
            "missing_reference_ids": json.dumps(sorted(expected_ids - computed_ids)),
            "distance_to_reference_hull_ev_per_atom": "",
            "energy_above_augmented_hull_ev_per_atom": "",
            "diagnostic_lower_bound_ehull_ev_per_atom": "",
            "decomposition_json": "",
            "hull_class": "blocked",
            "within_user_supplied_hull_threshold": False,
            "claim_scope": CLAIM_SCOPE,
            "static_settings_hash": candidate["settings_hash"],
        }
        if not blockers:
            from pymatgen.analysis.phase_diagram import PhaseDiagram

            reference_pd_entries = [_pd_entry(ref) for ref in usable_references]
            reference_diagram = PhaseDiagram(reference_pd_entries)
            assert candidate["energy_ev_per_atom"] is not None
            reference_hull_energy = float(
                reference_diagram.get_hull_energy_per_atom(candidate["composition"])
            )
            if not math.isfinite(reference_hull_energy):
                raise ValueError(
                    f"Non-finite reference hull energy for {candidate['entry_id']}"
                )
            distance = (
                float(candidate["energy_ev_per_atom"]) - reference_hull_energy
            )
            if not math.isfinite(distance):
                raise ValueError(
                    f"Non-finite reference hull distance for {candidate['entry_id']}"
                )
            competing_candidates = [
                other for other in converged_candidates
                if other["settings_hash"] == candidate["settings_hash"]
                and other["elements"].issubset(candidate["elements"])
            ]
            augmented_entries = reference_pd_entries + [
                _pd_entry(other) for other in competing_candidates
            ]
            augmented_diagram = PhaseDiagram(augmented_entries)
            decomposition, e_above = augmented_diagram.get_decomp_and_e_above_hull(
                _pd_entry(candidate), allow_negative=True
            )
            raw_e_above = float(e_above)
            if not math.isfinite(raw_e_above):
                raise ValueError(
                    f"Non-finite augmented hull energy for {candidate['entry_id']}"
                )
            if not decomposition:
                raise ValueError(
                    f"Empty hull decomposition for {candidate['entry_id']}"
                )
            decomposition_payload: dict[str, float] = {}
            for entry, amount in decomposition.items():
                coefficient = float(amount)
                if not math.isfinite(coefficient) or coefficient < -1e-10:
                    raise ValueError(
                        f"Invalid decomposition coefficient for "
                        f"{candidate['entry_id']}: {entry.name}={coefficient}"
                    )
                decomposition_payload[entry.name] = max(0.0, coefficient)
            e_above = max(0.0, raw_e_above)
            if e_above <= 1e-8:
                hull_class = "on_augmented_reference_hull"
            elif e_above <= near_hull_threshold_ev_per_atom:
                hull_class = "within_threshold_of_augmented_reference_hull"
            else:
                hull_class = "above_augmented_reference_hull"
            row.update({
                "distance_to_reference_hull_ev_per_atom": distance,
                "energy_above_augmented_hull_ev_per_atom": e_above,
                "decomposition_json": json.dumps(
                    decomposition_payload, sort_keys=True
                ),
                "hull_class": hull_class,
                "within_user_supplied_hull_threshold": (
                    e_above <= near_hull_threshold_ev_per_atom
                ),
            })
        elif diagnostic_incomplete and candidate["converged"]:
            lower_bound = _diagnostic_lower_bound(candidate, references)
            if lower_bound is not None:
                row["diagnostic_lower_bound_ehull_ev_per_atom"] = lower_bound
        result_rows.append(row)

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "hull_results.csv"
    _write_csv(results_path, result_rows)
    status_counts = Counter(row["coverage_status"] for row in result_rows)
    complete_count = status_counts.get("complete", 0)
    summary = {
        "workflow_version": WORKFLOW_VERSION,
        "status": (
            "complete" if complete_count == len(result_rows)
            else "blocked_incomplete_reference_coverage"
        ),
        "candidate_count": len(result_rows),
        "coverage_complete_hull_screen_count": complete_count,
        "blocked_candidate_count": len(result_rows) - complete_count,
        "near_hull_threshold_ev_per_atom": near_hull_threshold_ev_per_atom,
        "threshold_policy": (
            "User-supplied decision threshold; the boolean is not an independent "
            "thermodynamic-validation claim."
        ),
        "diagnostic_incomplete_enabled": diagnostic_incomplete,
        "coverage_status_counts": dict(status_counts),
        "claim_scope": CLAIM_SCOPE,
        "thermodynamically_validated_count": 0,
        "scientific_limit": (
            "This finite-smearing, non-spin-polarized, scalar-relativistic PBE "
            "screen is relative to one declared recomputed reference snapshot. "
            "It does not establish a magnetic/SOC ground state, experimental, "
            "oxidation, kinetic, finite-temperature, or service-condition stability."
        ),
        "candidate_entries": str(Path(candidate_entries_path).resolve()),
        "candidate_summary": str(Path(candidate_summary_path).resolve()),
        "candidate_entries_sha256": candidate_summary["results_manifest_sha256"],
        "candidate_energy_records_sha256": candidate_summary["energy_records_sha256"],
        "reference_entries": str(Path(reference_entries_path).resolve()),
        "reference_summary": str(Path(reference_summary_path).resolve()),
        "reference_entries_sha256": reference_summary["results_manifest_sha256"],
        "reference_energy_records_sha256": reference_summary["energy_records_sha256"],
        "reference_inventory": str(Path(reference_inventory_path).resolve()),
        "reference_inventory_sha256": coverage["inventory"]["reference_inventory_sha256"],
        "coverage_manifest": str(Path(coverage_manifest_path).resolve()),
        "results": str(results_path),
        "coverage_audit_by_system": audit_by_system,
        "candidates": result_rows,
    }
    summary_path = output_dir / "hull_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Hull workflow status: {summary['status']}")
    print(f"Coverage-complete screens: {complete_count}/{len(result_rows)}")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-entries", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--reference-entries", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--reference-inventory", type=Path, required=True)
    parser.add_argument("--coverage-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--near-hull-threshold-ev-per-atom", type=float, default=0.025
    )
    parser.add_argument("--diagnostic-incomplete", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    result = compute_hulls(
        candidate_entries_path=args.candidate_entries,
        candidate_summary_path=args.candidate_summary,
        reference_entries_path=args.reference_entries,
        reference_summary_path=args.reference_summary,
        reference_inventory_path=args.reference_inventory,
        coverage_manifest_path=args.coverage_manifest,
        output_dir=args.output_dir,
        near_hull_threshold_ev_per_atom=args.near_hull_threshold_ev_per_atom,
        diagnostic_incomplete=args.diagnostic_incomplete,
    )
    if result["status"] != "complete":
        sys.exit(2)
