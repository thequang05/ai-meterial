"""Audit a normalized thermal-label dataset before allowing model training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from pymatgen.core import Composition


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LABELS = ROOT / "research" / "phase_2" / "data" / "thermal" / "citrination_118353_v1" / "melting_by_reduced_formula.csv"
DEFAULT_LABEL_STATS = ROOT / "research" / "phase_2" / "data" / "thermal" / "citrination_118353_v1" / "dataset_stats.json"
DEFAULT_MATERIALS = ROOT / "research" / "phase_2" / "data" / "processed" / "materials.csv"
DEFAULT_OUTPUT = ROOT / "research" / "phase_2" / "reports" / "thermal_dataset_audit_citrination_v1"
DEFAULT_SPLIT = ROOT / "research" / "phase_2" / "data" / "splits" / "thermal_citrination_v1_seed42.json"

REFRACTORY = {"Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"}
REFRACTORY_CARBIDE = REFRACTORY | {"C"}


def reduced_formula(formula: str) -> str:
    return Composition(formula).reduced_formula


def deterministic_split(formulas: list[str], seed: int) -> dict[str, list[str]]:
    shuffled = sorted(formulas)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    test_count = max(1, round(count * 0.10))
    val_count = max(1, round(count * 0.10))
    return {
        "test": sorted(shuffled[:test_count]),
        "val": sorted(shuffled[test_count:test_count + val_count]),
        "train": sorted(shuffled[test_count + val_count:]),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit thermal labels before training.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--label-stats", type=Path, default=DEFAULT_LABEL_STATS)
    parser.add_argument("--materials", type=Path, default=DEFAULT_MATERIALS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    label_stats = json.loads(args.label_stats.read_text(encoding="utf-8"))
    with args.labels.open(newline="", encoding="utf-8") as handle:
        labels = list(csv.DictReader(handle))

    by_formula: defaultdict[str, list[dict]] = defaultdict(list)
    with args.materials.open(newline="", encoding="utf-8") as handle:
        for material in csv.DictReader(handle):
            try:
                by_formula[reduced_formula(material["formula"])].append(material)
            except (KeyError, TypeError, ValueError):
                continue

    matched, unmatched = [], []
    refractory_carbide_labels = []
    high_temperature_labels = []
    for label in labels:
        formula = label["formula_reduced"]
        composition = Composition(formula)
        elements = {str(element) for element in composition.elements}
        is_refractory_carbide = "C" in elements and bool(elements & REFRACTORY) and elements <= REFRACTORY_CARBIDE
        is_high_temperature = float(label["melting_point_K_median"]) >= 2000.0
        if is_refractory_carbide:
            refractory_carbide_labels.append(formula)
        if is_high_temperature:
            high_temperature_labels.append(formula)
        materials = by_formula.get(formula, [])
        base = {
            "formula_reduced": formula,
            "melting_point_K_median": label["melting_point_K_median"],
            "measurement_count": label["measurement_count"],
            "is_refractory_carbide": is_refractory_carbide,
            "is_high_temperature_ge_2000K": is_high_temperature,
        }
        if not materials:
            unmatched.append(base)
            continue
        for material in materials:
            matched.append(base | {
                "material_uid": material["uid"],
                "material_formula": material["formula"],
                "formation_energy_per_atom": material["formation_energy_per_atom"],
                "num_atoms": material["num_atoms"],
            })

    formulas = sorted(label["formula_reduced"] for label in labels)
    split = deterministic_split(formulas, args.seed)
    args.split_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.split_manifest.write_text(json.dumps({
        "dataset": "citrination_118353_v2_temporary_baseline",
        "label": "melting_point_K_median",
        "group_key": "reduced_formula",
        "seed": args.seed,
        "splits": split,
    }, indent=2) + "\n", encoding="utf-8")

    license_ok = label_stats.get("license_status", "").startswith("cc-by")
    train_gate = {
        "approved_for_production_thermal_training": False,
        "reasons": [
            "dataset_license_is_not_verified" if not license_ok else "temporary_dataset_only",
            "insufficient_refractory_carbide_coverage" if not refractory_carbide_labels else "limited_refractory_carbide_coverage",
            "dataset_is_too_small_for_a_reliable_thermal_evaluator",
        ],
    }
    summary = {
        "label_dataset": str(args.labels),
        "label_dataset_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "license_status": label_stats.get("license_status"),
        "label_count": len(labels),
        "matched_formula_count": len(labels) - len(unmatched),
        "unmatched_formula_count": len(unmatched),
        "matched_material_graph_count": len(matched),
        "refractory_carbide_formula_count": len(refractory_carbide_labels),
        "refractory_carbide_formulas": sorted(refractory_carbide_labels),
        "high_temperature_formula_count_ge_2000K": len(high_temperature_labels),
        "split_manifest": str(args.split_manifest),
        "split_counts": {name: len(values) for name, values in split.items()},
        "training_gate": train_gate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "matched_material_graphs.csv", matched)
    write_csv(args.output_dir / "unmatched_formulas.csv", unmatched)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
