"""Build a fixed competing-phase inventory from the local MP 2019 snapshot.

The selector includes every structure whose exact element set is one of the
non-empty subsystems required by the candidate chemical systems.  No
formation-energy cutoff or cell-size cutoff is allowed for official coverage.
An explicitly diagnostic per-subsystem cap is available, but it marks coverage
incomplete so ``compute_qe_hull.py`` cannot emit official results.

By default the command writes metadata/coverage only. ``--materialize-cifs``
copies the selected CIF text from the existing read-only SQLite structure
cache; it does not download data or run DFT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from compute_qe_hull import (
    _chemsys,
    _parse_entry,
    _read_csv,
    _subsystem_keys,
    _validate_snapshot_manifest_payload,
)


INVENTORY_VERSION = "mp2019_reference_inventory_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _receipt(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _source_energy(row: dict[str, str]) -> float:
    try:
        value = float(row.get("formation_energy_per_atom") or "nan")
    except ValueError:
        return math.inf
    return value if math.isfinite(value) else math.inf


def build_inventory(
    *,
    candidate_entries_path: Path,
    snapshot_manifest_path: Path,
    materials_csv_path: Path,
    structure_cache_path: Path,
    output_dir: Path,
    materialize_cifs: bool = False,
    diagnostic_max_per_subsystem: int | None = None,
) -> dict[str, Any]:
    if diagnostic_max_per_subsystem is not None and diagnostic_max_per_subsystem < 1:
        raise ValueError("diagnostic_max_per_subsystem must be positive")
    candidates = [
        _parse_entry(row, "candidate") for row in _read_csv(candidate_entries_path)
    ]
    converged = [candidate for candidate in candidates if candidate["converged"]]
    if not converged:
        raise ValueError("No dft_static_converged candidate entries were found")
    settings_hashes = {candidate["settings_hash"] for candidate in converged}
    if len(settings_hashes) != 1:
        raise ValueError("Candidate static settings are mixed")
    target_systems = {
        _chemsys(set(candidate["elements"])): set(candidate["elements"])
        for candidate in converged
    }
    required_subsystems = set().union(*(
        _subsystem_keys(elements) for elements in target_systems.values()
    ))

    materials_csv_path = Path(materials_csv_path).resolve()
    structure_cache_path = Path(structure_cache_path).resolve()
    snapshot_manifest_path = Path(snapshot_manifest_path).resolve()
    if not snapshot_manifest_path.is_file():
        raise FileNotFoundError(snapshot_manifest_path)
    if not materials_csv_path.is_file():
        raise FileNotFoundError(materials_csv_path)
    if not structure_cache_path.is_file():
        raise FileNotFoundError(structure_cache_path)
    snapshot_manifest = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
    _validate_snapshot_manifest_payload(snapshot_manifest)
    with materials_csv_path.open(newline="", encoding="utf-8") as handle:
        metadata_rows = list(csv.DictReader(handle))

    metadata_ids: set[str] = set()
    for index, row in enumerate(metadata_rows, start=2):
        source_id = str(row.get("source_id") or "").strip()
        formula = str(row.get("formula") or "").strip()
        elements = [
            item for item in str(row.get("elements_str") or "").split(";") if item
        ]
        if not source_id or not formula or not elements:
            raise ValueError(f"Malformed materials metadata row {index}")
        if source_id in metadata_ids:
            raise ValueError(f"Duplicate source_id in materials metadata: {source_id}")
        metadata_ids.add(source_id)

    by_subsystem: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metadata_rows:
        elements = {item for item in (row.get("elements_str") or "").split(";") if item}
        subsystem = _chemsys(elements)
        if subsystem in required_subsystems:
            by_subsystem[subsystem].append(row)

    selected_by_subsystem: dict[str, list[dict[str, str]]] = {}
    truncated_subsystems: set[str] = set()
    for subsystem in sorted(required_subsystems):
        rows = sorted(
            by_subsystem.get(subsystem, []),
            key=lambda row: (_source_energy(row), row.get("source_id") or ""),
        )
        if diagnostic_max_per_subsystem is not None and len(rows) > diagnostic_max_per_subsystem:
            rows = rows[:diagnostic_max_per_subsystem]
            truncated_subsystems.add(subsystem)
        selected_by_subsystem[subsystem] = rows

    print(
        f"Reference inventory plan: {len(target_systems)} candidate system(s), "
        f"{len(required_subsystems)} unique subsystem(s), "
        f"{sum(len(rows) for rows in selected_by_subsystem.values()):,} structure(s)"
    )

    conn = sqlite3.connect(f"file:{structure_cache_path}?mode=ro", uri=True)
    try:
        cache_ids = {
            str(row[0]) for row in conn.execute("SELECT material_id FROM structures")
        }
        cache_count = len(cache_ids)
        if cache_ids != metadata_ids:
            raise ValueError("Materials CSV and structure cache ID sets differ")
        common_id_digest = hashlib.sha256(
            "\n".join(sorted(metadata_ids)).encode("utf-8")
        ).hexdigest()
        materials_sha = _sha256(materials_csv_path)
        cache_sha = _sha256(structure_cache_path)
        if (
            snapshot_manifest.get("material_count") != len(metadata_ids)
            or snapshot_manifest.get("material_id_sha256") != common_id_digest
            or snapshot_manifest.get("materials_csv", {}).get("sha256") != materials_sha
            or snapshot_manifest.get("structure_cache", {}).get("sha256") != cache_sha
        ):
            raise ValueError(
                "Current metadata/cache do not match the locked MP snapshot manifest"
            )
        output_dir = Path(output_dir).resolve()
        cif_dir = output_dir / "cifs"
        output_dir.mkdir(parents=True, exist_ok=True)
        if materialize_cifs:
            cif_dir.mkdir(parents=True, exist_ok=True)

        inventory_rows: list[dict[str, Any]] = []
        entry_ids: set[str] = set()
        missing_cache_ids: set[str] = set()
        parse_error_ids: set[str] = set()
        subsystem_entry_ids: dict[str, list[str]] = {}
        for subsystem, rows in selected_by_subsystem.items():
            subsystem_entry_ids[subsystem] = []
            for source in rows:
                source_id = str(source["source_id"])
                entry_id = "ref_" + source_id.replace("-", "_")
                if entry_id in entry_ids:
                    raise ValueError(f"Duplicate reference entry ID: {entry_id}")
                entry_ids.add(entry_id)
                subsystem_entry_ids[subsystem].append(entry_id)
                cif_path = cif_dir / f"{entry_id}.cif"
                cache_present = source_id in cache_ids
                if not cache_present:
                    missing_cache_ids.add(source_id)
                if source.get("cif_parse_error"):
                    parse_error_ids.add(source_id)
                if materialize_cifs and cache_present:
                    record = conn.execute(
                        "SELECT cif FROM structures WHERE material_id = ?", (source_id,)
                    ).fetchone()
                    if not record or not record[0]:
                        missing_cache_ids.add(source_id)
                    else:
                        cif_path.write_text(str(record[0]), encoding="utf-8")
                source_cif_sha256 = _sha256(cif_path) if cif_path.is_file() else ""
                inventory_rows.append({
                    "entry_id": entry_id,
                    "entry_role": "reference",
                    "source_id": source_id,
                    "formula": source["formula"],
                    "subsystem": subsystem,
                    "source_formation_energy_per_atom": source.get(
                        "formation_energy_per_atom", ""
                    ),
                    "source_num_atoms": source.get("num_atoms", ""),
                    "cif_path": str(cif_path),
                    "cif_materialized": bool(materialize_cifs and cif_path.is_file()),
                    "source_cif_sha256": source_cif_sha256,
                    "source_cache_present": cache_present,
                    "source_cif_parse_error": source.get("cif_parse_error", ""),
                })
    finally:
        conn.close()

    inventory_digest = _receipt({
        "inventory_version": INVENTORY_VERSION,
        "materials_csv_sha256": materials_sha,
        "structure_cache_sha256": cache_sha,
        "selected": sorted([
            {
                "entry_id": row["entry_id"],
                "source_id": row["source_id"],
                "formula": row["formula"],
                "subsystem": row["subsystem"],
                "source_cif_sha256": row["source_cif_sha256"],
            }
            for row in inventory_rows
        ], key=lambda row: row["entry_id"]),
        "diagnostic_max_per_subsystem": diagnostic_max_per_subsystem,
    })

    inventory_path = output_dir / "reference_inventory.csv"
    fields = [
        "entry_id", "entry_role", "source_id", "formula", "subsystem",
        "source_formation_energy_per_atom", "source_num_atoms", "cif_path",
        "cif_materialized", "source_cif_sha256", "source_cache_present",
        "source_cif_parse_error",
    ]
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(inventory_rows)
    inventory_file_sha256 = _sha256(inventory_path)

    systems: dict[str, Any] = {}
    for system_key, elements in target_systems.items():
        subsystem_records: dict[str, Any] = {}
        system_complete = True
        for subsystem in sorted(_subsystem_keys(elements)):
            expected_ids = subsystem_entry_ids.get(subsystem, [])
            selected_sources = selected_by_subsystem.get(subsystem, [])
            selected_source_ids = {row["source_id"] for row in selected_sources}
            has_missing = bool(selected_source_ids & missing_cache_ids)
            has_parse_error = bool(selected_source_ids & parse_error_ids)
            if subsystem in truncated_subsystems:
                query_status = "diagnostic_truncated"
            elif not materialize_cifs:
                query_status = "cifs_not_materialized"
            elif has_missing:
                query_status = "source_structure_missing"
            elif has_parse_error:
                query_status = "source_cif_parse_error"
            else:
                query_status = "complete"
            if query_status != "complete":
                system_complete = False
            subsystem_records[subsystem] = {
                "query_status": query_status,
                "query_receipt_sha256": _receipt({
                    "inventory_sha256": inventory_digest,
                    "subsystem": subsystem,
                    "expected_reference_ids": expected_ids,
                    "query_status": query_status,
                }),
                "expected_reference_ids": expected_ids,
            }
        systems[system_key] = {
            "elements": sorted(elements),
            "subsystems": subsystem_records,
            "deduplications": [],
            "coverage_complete_assertion": system_complete,
        }

    coverage = {
        "schema_version": "dft_reference_coverage_v1",
        "coverage_id": f"mp2019_{inventory_digest[:16]}",
        "static_settings_hash": next(iter(settings_hashes)),
        "inventory": {
            "source_name": "Materials Project local mp.2019.04.01 snapshot",
            "source_snapshot_id": "mp.2019.04.01",
            "inventory_sha256": inventory_digest,
            "query_code_version": INVENTORY_VERSION,
            "scope": "all structures from every non-empty subsystem",
            "materials_csv": str(materials_csv_path),
            "materials_csv_sha256": materials_sha,
            "structure_cache": str(structure_cache_path),
            "structure_cache_sha256": cache_sha,
            "structure_cache_count": cache_count,
            "snapshot_manifest": str(snapshot_manifest_path),
            "snapshot_manifest_sha256": _sha256(snapshot_manifest_path),
            "reference_inventory": str(inventory_path),
            "reference_inventory_sha256": inventory_file_sha256,
            "diagnostic_max_per_subsystem": diagnostic_max_per_subsystem,
        },
        "systems": systems,
    }
    coverage_path = output_dir / "reference_coverage_v1.json"
    coverage_path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    summary = {
        "inventory_version": INVENTORY_VERSION,
        "reference_count": len(inventory_rows),
        "subsystem_count": len(required_subsystems),
        "system_count": len(target_systems),
        "cifs_materialized": materialize_cifs,
        "missing_structure_count": len(missing_cache_ids),
        "source_cif_parse_error_count": len(parse_error_ids),
        "diagnostic_truncated_subsystems": sorted(truncated_subsystems),
        "inventory": str(inventory_path),
        "inventory_file_sha256": inventory_file_sha256,
        "snapshot_manifest": str(snapshot_manifest_path),
        "coverage_manifest": str(coverage_path),
        "dft_calculation_started": False,
    }
    summary_path = output_dir / "reference_inventory_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Reference inventory: {len(inventory_rows)} entries")
    print(f"Inventory: {inventory_path}")
    print(f"Coverage:  {coverage_path}")
    print("No DFT calculation was started.")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-entries", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--materials-csv", type=Path, required=True)
    parser.add_argument("--structure-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--materialize-cifs", action="store_true")
    parser.add_argument("--diagnostic-max-per-subsystem", type=int)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    build_inventory(
        candidate_entries_path=args.candidate_entries,
        snapshot_manifest_path=args.snapshot_manifest,
        materials_csv_path=args.materials_csv,
        structure_cache_path=args.structure_cache,
        output_dir=args.output_dir,
        materialize_cifs=args.materialize_cifs,
        diagnostic_max_per_subsystem=args.diagnostic_max_per_subsystem,
    )
