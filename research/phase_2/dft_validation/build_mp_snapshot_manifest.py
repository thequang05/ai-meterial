"""Verify and lock the local MP-2019 raw/metadata/structure-cache snapshot.

This code-only utility streams the large raw JSON (it does not load 3.8 GB into
RAM), hashes all three artifacts, and requires exact material-ID equality among
the raw JSON, processed metadata CSV, and SQLite structure cache.  It performs
no network access and no DFT calculation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import warnings
from pathlib import Path
from typing import Any

from pymatgen.core import Composition, Structure


SCHEMA_VERSION = "mp_snapshot_manifest_v1"
RAW_ID_PATTERN = re.compile(rb'"material_id"\s*:\s*"(mp-\d+)"')


def _id_digest(ids: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _hash_and_raw_ids(path: Path) -> tuple[str, set[str], int]:
    digest = hashlib.sha256()
    ids: set[str] = set()
    occurrences = 0
    overlap = 256
    carry = b""
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            data = carry + chunk
            safe_end = max(0, len(data) - overlap)
            for match in RAW_ID_PATTERN.finditer(data):
                if match.start() < safe_end:
                    occurrences += 1
                    ids.add(match.group(1).decode("ascii"))
            carry = data[safe_end:]
    for match in RAW_ID_PATTERN.finditer(carry):
        occurrences += 1
        ids.add(match.group(1).decode("ascii"))
    return digest.hexdigest(), ids, occurrences


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *, raw_json_path: Path, materials_csv_path: Path,
    structure_cache_path: Path, output_path: Path,
) -> dict[str, Any]:
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing MP snapshot manifest: "
            f"{output_path}"
        )
    raw_json_path = Path(raw_json_path).resolve()
    materials_csv_path = Path(materials_csv_path).resolve()
    structure_cache_path = Path(structure_cache_path).resolve()
    for path in (raw_json_path, materials_csv_path, structure_cache_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    raw_sha, raw_ids, raw_occurrences = _hash_and_raw_ids(raw_json_path)
    if raw_occurrences != len(raw_ids):
        raise ValueError("Raw snapshot contains duplicate material_id values")

    with materials_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    csv_ids: set[str] = set()
    csv_compositions: dict[str, tuple[str, frozenset[str]]] = {}
    for index, row in enumerate(rows, start=2):
        source_id = str(row.get("source_id") or "").strip()
        uid = str(row.get("uid") or "").strip()
        formula = str(row.get("formula") or "").strip()
        elements = [item for item in str(row.get("elements_str") or "").split(";") if item]
        if not source_id or not formula or not elements:
            raise ValueError(f"Malformed materials CSV row {index}")
        if uid != f"MP_{source_id}":
            raise ValueError(f"UID/source_id mismatch at CSV row {index}")
        if source_id in csv_ids:
            raise ValueError(f"Duplicate CSV source_id: {source_id}")
        csv_ids.add(source_id)
        csv_compositions[source_id] = (
            Composition(formula).reduced_formula,
            frozenset(elements),
        )

    conn = sqlite3.connect(f"file:{structure_cache_path}?mode=ro", uri=True)
    try:
        cache_cursor = conn.execute(
            "SELECT material_id, cif FROM structures ORDER BY material_id"
        )
        cache_ids: set[str] = set()
        composition_audit = hashlib.sha256()
        for material_id, cif in cache_cursor:
            material_id = str(material_id)
            if material_id in cache_ids:
                raise ValueError(f"Duplicate cache material_id: {material_id}")
            if not str(cif or "").strip():
                raise ValueError(f"Empty cached CIF for {material_id}")
            expected = csv_compositions.get(material_id)
            if expected is None:
                raise ValueError(f"Cache ID absent from metadata: {material_id}")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    structure = Structure.from_str(str(cif), fmt="cif")
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Cached CIF cannot be parsed: {material_id}") from exc
            actual_formula = structure.composition.reduced_formula
            actual_elements = frozenset(
                element.symbol for element in structure.composition.elements
            )
            if (actual_formula, actual_elements) != expected:
                raise ValueError(
                    f"Metadata/CIF composition mismatch for {material_id}: "
                    f"metadata={expected}, cif={(actual_formula, actual_elements)}"
                )
            composition_audit.update(json.dumps({
                "material_id": material_id,
                "formula": actual_formula,
                "elements": sorted(actual_elements),
                "cif_sha256": hashlib.sha256(str(cif).encode("utf-8")).hexdigest(),
            }, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            composition_audit.update(b"\n")
            cache_ids.add(material_id)
    finally:
        conn.close()

    if raw_ids != csv_ids or raw_ids != cache_ids:
        raise ValueError(json.dumps({
            "raw_only_vs_csv": sorted(raw_ids - csv_ids)[:20],
            "csv_only_vs_raw": sorted(csv_ids - raw_ids)[:20],
            "raw_only_vs_cache": sorted(raw_ids - cache_ids)[:20],
            "cache_only_vs_raw": sorted(cache_ids - raw_ids)[:20],
        }))
    common_id_digest = _id_digest(raw_ids)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_exact_id_sets",
        "snapshot_id": "mp.2019.04.01",
        "material_count": len(raw_ids),
        "material_id_sha256": common_id_digest,
        "composition_audit_sha256": composition_audit.hexdigest(),
        "composition_audit_status": "all_cache_cifs_match_metadata",
        "raw_json": {
            "path": str(raw_json_path),
            "sha256": raw_sha,
            "material_count": len(raw_ids),
            "material_id_occurrences": raw_occurrences,
            "material_id_sha256": common_id_digest,
        },
        "materials_csv": {
            "path": str(materials_csv_path),
            "sha256": _hash(materials_csv_path),
            "material_count": len(csv_ids),
            "material_id_sha256": _id_digest(csv_ids),
        },
        "structure_cache": {
            "path": str(structure_cache_path),
            "sha256": _hash(structure_cache_path),
            "material_count": len(cache_ids),
            "material_id_sha256": _id_digest(cache_ids),
            "composition_audit_sha256": composition_audit.hexdigest(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Verified exact MP snapshot IDs: {len(raw_ids):,}")
    print(f"Manifest: {output_path}")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-json", type=Path, required=True)
    parser.add_argument("--materials-csv", type=Path, required=True)
    parser.add_argument("--structure-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    build_manifest(
        raw_json_path=args.raw_json,
        materials_csv_path=args.materials_csv,
        structure_cache_path=args.structure_cache,
        output_path=args.output,
    )
