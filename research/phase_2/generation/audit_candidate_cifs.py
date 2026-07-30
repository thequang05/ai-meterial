"""Audit generated candidate CIFs before structural relaxation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pymatgen.core import Structure

from structure_audit import (
    AUDIT_VERSION,
    STRUCTURE_MATCHER_SETTINGS,
    audit_cif,
    find_duplicate_structures,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _resolve_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path
    return manifest_path.parent / path


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def audit_manifest(
    manifest_path: Path,
    output_dir: Path,
    *,
    stage: str = "pre_relax",
) -> dict[str, Any]:
    if stage not in {"pre_relax", "post_relax"}:
        raise ValueError(f"Unsupported audit stage: {stage}")
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        input_rows = list(csv.DictReader(handle))

    results: list[dict[str, Any]] = []
    parsed: list[tuple[str, Structure]] = []
    for row in input_rows:
        candidate_id = row.get("candidate_id", "")
        expected_formula = row.get("candidate_formula") or row.get("formula") or ""
        cif_value = row.get("cif_path") or row.get("output_cif") or ""
        cif_path = _resolve_path(cif_value, manifest_path)
        result = audit_cif(cif_path, expected_formula=expected_formula)
        result.update({
            "candidate_id": candidate_id,
            "prototype_uid": row.get("prototype_uid", ""),
            "input_structure_status": row.get("structure_status", ""),
        })
        results.append(result)
        if result["audit_status"] != "fail":
            try:
                parsed.append((candidate_id, Structure.from_file(cif_path)))
            except Exception:  # already represented by audit_cif if reproducible
                pass

    duplicate_map = find_duplicate_structures(parsed)
    for result in results:
        duplicate_of = duplicate_map.get(result["candidate_id"], "")
        result["duplicate_of"] = duplicate_of
        if duplicate_of and result["audit_status"] != "fail":
            result["warning_reasons"].append(f"duplicate_structure_of:{duplicate_of}")
            result["audit_status"] = "warn"

    csv_path = output_dir / "structure_audit.csv"
    preferred_fields = [
        "candidate_id", "prototype_uid", "expected_formula", "actual_formula",
        "input_structure_status", "audit_status", "failure_reasons", "warning_reasons",
        "duplicate_of", "num_sites", "volume_a3", "volume_per_atom_a3",
        "density_g_cm3", "min_distance_angstrom", "min_distance_pair",
        "min_radius_ratio", "min_radius_ratio_pair", "space_group_symbol",
        "space_group_number", "is_ordered", "cif_path", "parser_warnings",
    ]
    extra_fields = sorted({key for row in results for key in row} - set(preferred_fields))
    fields = preferred_fields + extra_fields
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _csv_value(row.get(key, "")) for key in fields} for row in results)

    status_counts = Counter(result["audit_status"] for result in results)
    geometry_ready_count = sum(
        result["audit_status"] != "fail" for result in results
    )
    duplicate_count = sum(bool(result.get("duplicate_of")) for result in results)
    unique_ready_count = sum(
        result["audit_status"] != "fail" and not result.get("duplicate_of")
        for result in results
    )
    summary = {
        "audit_version": AUDIT_VERSION,
        "audit_stage": stage,
        "manifest": str(manifest_path),
        "candidate_count": len(results),
        "status_counts": dict(status_counts),
        "geometry_ready_count": geometry_ready_count,
        "duplicate_structure_count": duplicate_count,
        "structurally_unique_ready_count": unique_ready_count,
        "structure_matcher_settings": STRUCTURE_MATCHER_SETTINGS,
        # Kept as the downstream-facing count for compatibility.  Structural
        # duplicates are warnings, but they must not consume relaxation budget.
        "ready_for_relaxation_count": unique_ready_count,
        "important_limit": (
            "This is a geometry sanity audit of "
            + (
                "unrelaxed prototype-substitution CIFs"
                if stage == "pre_relax"
                else "machine-learning-relaxed CIFs"
            )
            + "; it is not DFT validation or evidence of thermodynamic/thermal stability."
        ),
        "candidates": results,
    }
    json_path = output_dir / "structure_audit_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Structure geometry audit stage={stage} ({AUDIT_VERSION})")
    for result in results:
        distance = result.get("min_distance_angstrom")
        distance_text = f"{distance:.3f} A" if isinstance(distance, (int, float)) else "n/a"
        print(
            f"- {result['candidate_id']}: {result.get('actual_formula', '?')} "
            f"status={result['audit_status']} min_distance={distance_text} "
            f"space_group={result.get('space_group_symbol', 'n/a')}"
        )
        for reason in result.get("failure_reasons", []):
            print(f"    FAIL: {reason}")
        for reason in result.get("warning_reasons", []):
            print(f"    WARN: {reason}")
    print(f"\nGeometry-ready: {summary['geometry_ready_count']}/{len(results)}")
    print(
        "Structurally unique and ready for relaxation: "
        f"{summary['structurally_unique_ready_count']}/{len(results)} "
        f"(duplicates excluded: {summary['duplicate_structure_count']})"
    )
    print(f"CSV:     {csv_path}")
    print(f"Summary: {json_path}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["pre_relax", "post_relax"],
        default="pre_relax",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    audit_manifest(args.manifest, args.output_dir, stage=args.stage)
