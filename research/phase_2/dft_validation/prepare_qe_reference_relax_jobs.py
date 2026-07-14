"""Prepare audited QE ``vc-relax`` jobs from a fixed reference inventory CSV.

This is a thin, non-executing adapter around ``prepare_qe_jobs.py``.  It keeps
reference entry IDs/roles intact so the same relaxation, static-SCF, and
collection tools can later produce ``reference_static_energies.csv`` for the
coverage-gated hull workflow.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from prepare_qe_jobs import prepare_jobs


SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_inventory(path: Path) -> list[dict[str, str]]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Reference inventory is empty")
    return rows


def prepare_reference_relax_jobs(
    *,
    reference_inventory_path: Path,
    config_path: Path,
    pseudo_manifest_path: Path,
    pseudo_dir: Path,
    output_dir: Path,
    pw_executable: str = "pw.x",
) -> dict[str, Any]:
    rows = _read_inventory(reference_inventory_path)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_snapshot = output_dir / "reference_inventory_snapshot.csv"
    shutil.copy2(Path(reference_inventory_path).resolve(), inventory_snapshot)

    seen: set[str] = set()
    queue: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if row.get("entry_role") != "reference":
            raise ValueError(f"Inventory row {index} is not entry_role=reference")
        if str(row.get("cif_materialized") or "").lower() not in {"true", "1", "yes"}:
            raise ValueError(f"Reference CIF was not materialized at row {index}")
        entry_id = str(row.get("entry_id") or "").strip()
        if not SAFE_ID.fullmatch(entry_id):
            raise ValueError(
                f"Reference entry_id must be path-safe and <=128 chars: {entry_id!r}"
            )
        if entry_id in seen:
            raise ValueError(f"Duplicate reference entry_id: {entry_id}")
        seen.add(entry_id)
        formula = str(row.get("formula") or "").strip()
        cif_path = str(row.get("cif_path") or "").strip()
        if not formula or not cif_path:
            raise ValueError(f"Reference {entry_id} requires formula and cif_path")
        resolved_cif = Path(cif_path).expanduser().resolve()
        expected_cif_sha = str(row.get("source_cif_sha256") or "").lower()
        if not resolved_cif.is_file():
            raise FileNotFoundError(resolved_cif)
        if not re.fullmatch(r"[0-9a-f]{64}", expected_cif_sha):
            raise ValueError(f"Reference {entry_id} has no valid source CIF hash")
        if _sha256(resolved_cif) != expected_cif_sha:
            raise ValueError(f"Reference source CIF hash mismatch: {entry_id}")
        queue.append({
            "rank": index,
            "entry_id": entry_id,
            "entry_role": "reference",
            "source_id": str(row.get("source_id") or entry_id),
            # The generic runner's historical field is candidate_id. It is a
            # job identifier here; entry_role preserves the scientific role.
            "candidate_id": entry_id,
            "formula": formula,
            "relaxed_cif": str(resolved_cif),
            "source_cif_sha256": expected_cif_sha,
            "subsystem": str(row.get("subsystem") or ""),
        })
    report = {
        "status": "reference_inventory_complete",
        "source_inventory": str(inventory_snapshot),
        "source_inventory_sha256": _sha256(inventory_snapshot),
        "recommended_dft_queue": queue,
        "scientific_limit": (
            "Inventory membership is not a DFT result. Every reference phase "
            "must pass vc-relax and static-SCF with the locked common settings."
        ),
    }
    report_path = output_dir / "reference_relaxation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return prepare_jobs(
        validation_report_path=report_path,
        config_path=config_path,
        output_dir=output_dir,
        top_n=len(queue),
        pseudo_manifest_path=pseudo_manifest_path,
        pseudo_dir=pseudo_dir,
        pw_executable=pw_executable,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-inventory", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pseudo-manifest", type=Path, required=True)
    parser.add_argument("--pseudo-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pw-executable", default="pw.x")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    prepare_reference_relax_jobs(
        reference_inventory_path=args.reference_inventory,
        config_path=args.config,
        pseudo_manifest_path=args.pseudo_manifest,
        pseudo_dir=args.pseudo_dir,
        output_dir=args.output_dir,
        pw_executable=args.pw_executable,
    )
