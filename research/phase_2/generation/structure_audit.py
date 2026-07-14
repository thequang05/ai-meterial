"""Fast, non-relaxing geometry checks for generated candidate structures.

This module deliberately does not estimate stability or thermal performance.
It checks whether a CIF is internally consistent before or after an external
relaxation workflow.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Element, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


AUDIT_VERSION = "candidate_structure_geometry_v2"
STRUCTURE_MATCHER_SETTINGS = {"ltol": 0.2, "stol": 0.3, "angle_tol": 5.0}


@dataclass(frozen=True)
class StructureAuditThresholds:
    """Conservative geometry thresholds for a structure sanity gate."""

    fail_min_distance_angstrom: float = 0.75
    warn_min_distance_angstrom: float = 1.00
    fail_min_radius_ratio: float = 0.45
    warn_min_radius_ratio: float = 0.60
    fail_min_volume_per_atom_a3: float = 2.0
    fail_max_volume_per_atom_a3: float = 200.0
    warn_min_volume_per_atom_a3: float = 5.0
    warn_max_volume_per_atom_a3: float = 100.0
    warn_max_density_g_cm3: float = 30.0


DEFAULT_THRESHOLDS = StructureAuditThresholds()


def _as_float(value: Any) -> float:
    """Convert pymatgen's unit-bearing values to a plain float."""
    return float(value)


def _atomic_radius(symbol: str) -> float | None:
    element = Element(symbol)
    radius = element.atomic_radius or element.atomic_radius_calculated
    if radius is None:
        return None
    value = _as_float(radius)
    return value if math.isfinite(value) and value > 0 else None


def _minimum_pair_metrics(structure: Structure) -> dict[str, Any]:
    """Return the closest distinct-site pair and radius-normalized distance."""
    if len(structure) < 2:
        return {
            "min_distance_angstrom": None,
            "min_distance_pair": "",
            "min_radius_ratio": None,
            "min_radius_ratio_pair": "",
        }

    distances = structure.distance_matrix
    min_distance = math.inf
    min_distance_pair = ""
    min_ratio = math.inf
    min_ratio_pair = ""
    for left in range(len(structure)):
        left_symbol = structure[left].specie.symbol
        left_radius = _atomic_radius(left_symbol)
        for right in range(left + 1, len(structure)):
            right_symbol = structure[right].specie.symbol
            distance = float(distances[left, right])
            pair = f"{left}:{left_symbol}-{right}:{right_symbol}"
            if distance < min_distance:
                min_distance = distance
                min_distance_pair = pair

            right_radius = _atomic_radius(right_symbol)
            if left_radius is None or right_radius is None:
                continue
            ratio = distance / (left_radius + right_radius)
            if ratio < min_ratio:
                min_ratio = ratio
                min_ratio_pair = pair

    return {
        "min_distance_angstrom": min_distance,
        "min_distance_pair": min_distance_pair,
        "min_radius_ratio": None if min_ratio == math.inf else min_ratio,
        "min_radius_ratio_pair": min_ratio_pair,
    }


def audit_structure(
    structure: Structure,
    *,
    expected_formula: str | None = None,
    thresholds: StructureAuditThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Audit one parsed structure without changing or relaxing it."""
    failures: list[str] = []
    audit_warnings: list[str] = []
    actual_formula = structure.composition.reduced_formula

    expected_reduced = ""
    if expected_formula:
        try:
            expected_reduced = Composition(expected_formula).reduced_formula
        except (TypeError, ValueError):
            failures.append("invalid_expected_formula")
        else:
            if actual_formula != expected_reduced:
                failures.append(
                    f"formula_mismatch:expected={expected_reduced}:actual={actual_formula}"
                )

    if not structure.is_ordered:
        failures.append("partial_or_disordered_occupancy")
    if len(structure) == 0:
        failures.append("empty_structure")

    volume = float(structure.volume)
    volume_per_atom = volume / len(structure) if len(structure) else math.nan
    density = _as_float(structure.density) if len(structure) else math.nan
    if not math.isfinite(volume) or volume <= 0:
        failures.append("invalid_cell_volume")
    if not math.isfinite(volume_per_atom):
        failures.append("invalid_volume_per_atom")
    elif (
        volume_per_atom < thresholds.fail_min_volume_per_atom_a3
        or volume_per_atom > thresholds.fail_max_volume_per_atom_a3
    ):
        failures.append(f"implausible_volume_per_atom:{volume_per_atom:.3f}")
    elif (
        volume_per_atom < thresholds.warn_min_volume_per_atom_a3
        or volume_per_atom > thresholds.warn_max_volume_per_atom_a3
    ):
        audit_warnings.append(f"unusual_volume_per_atom:{volume_per_atom:.3f}")

    if not math.isfinite(density) or density <= 0:
        failures.append("invalid_density")
    elif density > thresholds.warn_max_density_g_cm3:
        audit_warnings.append(f"unusually_high_density:{density:.3f}")

    pair_metrics = _minimum_pair_metrics(structure)
    min_distance = pair_metrics["min_distance_angstrom"]
    min_ratio = pair_metrics["min_radius_ratio"]
    if min_distance is not None:
        if min_distance < thresholds.fail_min_distance_angstrom:
            failures.append(f"atomic_overlap:min_distance={min_distance:.3f}")
        elif min_distance < thresholds.warn_min_distance_angstrom:
            audit_warnings.append(f"very_short_distance:{min_distance:.3f}")
    if min_ratio is not None:
        if min_ratio < thresholds.fail_min_radius_ratio:
            failures.append(f"atomic_radius_overlap:ratio={min_ratio:.3f}")
        elif min_ratio < thresholds.warn_min_radius_ratio:
            audit_warnings.append(f"short_radius_normalized_distance:{min_ratio:.3f}")

    space_group_symbol = ""
    space_group_number: int | None = None
    try:
        analyzer = SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5.0)
        space_group_symbol = analyzer.get_space_group_symbol()
        space_group_number = analyzer.get_space_group_number()
    except Exception as exc:  # noqa: BLE001
        audit_warnings.append(f"symmetry_analysis_failed:{type(exc).__name__}")

    audit_status = "fail" if failures else ("warn" if audit_warnings else "pass")
    return {
        "audit_version": AUDIT_VERSION,
        "audit_status": audit_status,
        "failure_reasons": failures,
        "warning_reasons": audit_warnings,
        "expected_formula": expected_reduced,
        "actual_formula": actual_formula,
        "num_sites": len(structure),
        "is_ordered": bool(structure.is_ordered),
        "volume_a3": volume,
        "volume_per_atom_a3": volume_per_atom,
        "density_g_cm3": density,
        **pair_metrics,
        "lattice_a_angstrom": float(structure.lattice.a),
        "lattice_b_angstrom": float(structure.lattice.b),
        "lattice_c_angstrom": float(structure.lattice.c),
        "lattice_alpha_deg": float(structure.lattice.alpha),
        "lattice_beta_deg": float(structure.lattice.beta),
        "lattice_gamma_deg": float(structure.lattice.gamma),
        "space_group_symbol": space_group_symbol,
        "space_group_number": space_group_number,
        "thresholds": asdict(thresholds),
    }


def audit_cif(
    cif_path: Path,
    *,
    expected_formula: str | None = None,
    thresholds: StructureAuditThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Parse and audit one CIF, retaining non-fatal parser warnings."""
    cif_path = Path(cif_path)
    if not cif_path.exists():
        return {
            "audit_version": AUDIT_VERSION,
            "audit_status": "fail",
            "failure_reasons": ["cif_not_found"],
            "warning_reasons": [],
            "cif_path": str(cif_path),
            "parser_warnings": [],
        }

    parser_warnings: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            structure = Structure.from_file(cif_path)
        parser_warnings = [str(item.message) for item in caught]
    except Exception as exc:  # noqa: BLE001
        return {
            "audit_version": AUDIT_VERSION,
            "audit_status": "fail",
            "failure_reasons": [f"cif_parse_failed:{type(exc).__name__}:{exc}"],
            "warning_reasons": [],
            "cif_path": str(cif_path),
            "parser_warnings": parser_warnings,
        }

    result = audit_structure(
        structure,
        expected_formula=expected_formula,
        thresholds=thresholds,
    )
    result["cif_path"] = str(cif_path)
    result["parser_warnings"] = parser_warnings
    return result


def find_duplicate_structures(
    structures: list[tuple[str, Structure]],
) -> dict[str, str]:
    """Return candidate -> first equivalent candidate within the audit batch."""
    matcher = StructureMatcher(**STRUCTURE_MATCHER_SETTINGS)
    duplicates: dict[str, str] = {}
    for index, (candidate_id, structure) in enumerate(structures):
        for prior_id, prior_structure in structures[:index]:
            if structure.composition.reduced_composition != prior_structure.composition.reduced_composition:
                continue
            if matcher.fit(prior_structure, structure):
                duplicates[candidate_id] = prior_id
                break
    return duplicates
