"""Normalize the temporary Citrination 118353 melting-point export.

This source is retained as a provenance-preserving baseline only.  The
Citrination dataset page does not expose an explicit reuse license, so the
output records that status and must not be presented as an open training
corpus or merged with CC-BY sources without a later rights review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from pymatgen.core import Composition


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = ROOT / "data" / "tr_merged.json"
DEFAULT_OUTPUT = ROOT / "research" / "phase_2" / "data" / "thermal" / "citrination_118353_v1"
SOURCE_URL = "https://citrination.com/datasets/118353/show_files/"
LICENSE_STATUS = "unverified_citrination_dataset_license"


def to_kelvin(value: str, unit: str) -> float:
    number = float(str(value).replace(",", ""))
    normalized = unit.strip().replace(" ", "")
    if normalized == "K":
        return number
    if normalized in {"$^{\\circ}$C", "°C", "degC", "C"}:
        return number + 273.15
    raise ValueError(f"unsupported_temperature_unit:{unit}")


def citation_text(references: list[dict]) -> str:
    return " | ".join(
        str(reference.get("citation", "")).strip()
        for reference in references
        if str(reference.get("citation", "")).strip()
    )


def prepare(input_path: Path, output_dir: Path) -> dict:
    source_bytes = input_path.read_bytes()
    payload = json.loads(source_bytes)
    if not isinstance(payload, list):
        raise ValueError("expected_top_level_json_list")

    raw_rows: list[dict] = []
    rejected: defaultdict[str, int] = defaultdict(int)
    for entry_index, entry in enumerate(payload):
        raw_formula = str(entry.get("chemicalFormula", "")).strip()
        try:
            reduced_formula = Composition(raw_formula).reduced_formula
        except (ValueError, TypeError):
            rejected["invalid_formula"] += 1
            continue
        for property_index, property_data in enumerate(entry.get("properties", [])):
            if property_data.get("name") != "Melting point":
                continue
            unit = str(property_data.get("units", "")).strip()
            reference = citation_text(property_data.get("references", []))
            for scalar_index, scalar in enumerate(property_data.get("scalars", [])):
                try:
                    value = float(str(scalar["value"]).replace(",", ""))
                    value_k = to_kelvin(str(scalar["value"]), unit)
                except (KeyError, TypeError, ValueError) as exc:
                    rejected[str(exc)] += 1
                    continue
                raw_rows.append({
                    "source_dataset": "citrination_118353_v2",
                    "source_url": SOURCE_URL,
                    "license_status": LICENSE_STATUS,
                    "entry_index": entry_index,
                    "property_index": property_index,
                    "scalar_index": scalar_index,
                    "formula_raw": raw_formula,
                    "formula_reduced": reduced_formula,
                    "property_name": "melting_point",
                    "value_raw": value,
                    "unit_raw": unit,
                    "melting_point_K": round(value_k, 4),
                    "data_type": property_data.get("dataType", ""),
                    "citation": reference,
                })

    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for row in raw_rows:
        grouped[row["formula_reduced"]].append(row)
    clean_rows = []
    for formula in sorted(grouped):
        rows = grouped[formula]
        values = [row["melting_point_K"] for row in rows]
        citations = sorted({row["citation"] for row in rows if row["citation"]})
        clean_rows.append({
            "source_dataset": "citrination_118353_v2",
            "source_url": SOURCE_URL,
            "license_status": LICENSE_STATUS,
            "formula_reduced": formula,
            "property_name": "melting_point",
            "melting_point_K_median": round(median(values), 4),
            "melting_point_K_min": round(min(values), 4),
            "melting_point_K_max": round(max(values), 4),
            "measurement_count": len(values),
            "citations": " | ".join(citations),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "melting_raw_measurements.csv"
    clean_path = output_dir / "melting_by_reduced_formula.csv"
    for path, rows in ((raw_path, raw_rows), (clean_path, clean_rows)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)

    stats = {
        "dataset_version": "citrination_118353_v2",
        "source_url": SOURCE_URL,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "license_status": LICENSE_STATUS,
        "input_entries": len(payload),
        "raw_melting_measurements": len(raw_rows),
        "unique_reduced_formulas": len(clean_rows),
        "rejected_records": dict(sorted(rejected.items())),
        "usage": "temporary_baseline_only",
        "not_sufficient_for": "a production refractory-carbide thermal evaluator",
    }
    (output_dir / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Citrination 118353 melting-point export.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    stats = prepare(args.input, args.output_dir)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
