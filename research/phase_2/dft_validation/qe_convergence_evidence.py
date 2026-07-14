"""Semantic verification for provisional QE convergence evidence.

The convergence collector intentionally writes the same per-point evidence to
both JSON and CSV.  Hashing those files independently is not sufficient: a
later stage must also prove that they describe the same measurements and that
the reported stable-tail selections follow from those measurements.  This
module is shared by confirmation preparation and certificate verification so
the two trust boundaries cannot silently diverge.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from typing import Any, Iterable


_FLOAT_FIELDS = {
    "num_atoms",
    "cutoff_pair_multiplier",
    "requested_kpoint_spacing_inv_angstrom",
    "ecutwfc_ry",
    "ecutrho_ry",
    "total_energy_ry",
    "energy_ev_per_atom",
    "max_force_delta_to_anchor_ev_per_angstrom",
    "max_stress_delta_to_anchor_kbar",
    "energy_delta_to_anchor_mev_per_atom",
}
_INT_FIELDS = {
    "rank", "level_index", "cutoff_level_index", "kpoint_level_index",
}
_BOOL_FIELDS = {"point_within_tolerances", "stable_tail_from_this_level"}
_JSON_FIELDS = {
    "force_components_ev_per_angstrom",
    "stress_components_kbar",
    "gate_failures",
    "execution_provenance",
}

PRODUCTION_TRANSFER_SCOPE = (
    "Conservative transfer at the strictest jointly tested sweep anchor. "
    "This does not prove convergence for each unseen candidate or reference "
    "structure."
)
RY_TO_EV = 13.605693122994


def derive_strictest_tested_settings(
    *, protocol: dict[str, Any], base_ecutwfc_ry: Any, base_ecutrho_ry: Any,
) -> dict[str, float]:
    """Return the conservative joint anchor at the edge of the tested window.

    This is a transfer policy, not evidence that every unseen structure is
    individually converged.  It uses the largest tested cutoff multiplier and
    the smallest (densest) tested k-point spacing, which the sweep protocol
    requires to coincide at each representative's anchor calculation.
    """

    if not isinstance(protocol, dict):
        raise ValueError("Locked convergence protocol is missing")
    raw_multipliers = protocol.get("cutoff_pair_multipliers")
    raw_spacings = protocol.get("kpoint_spacings_inv_angstrom")
    if not isinstance(raw_multipliers, list) or not raw_multipliers:
        raise ValueError("Locked cutoff sweep window is missing")
    if not isinstance(raw_spacings, list) or not raw_spacings:
        raise ValueError("Locked k-point sweep window is missing")
    multipliers = [
        _finite_float(value, "cutoff_pair_multipliers") for value in raw_multipliers
    ]
    spacings = [
        _finite_float(value, "kpoint_spacings_inv_angstrom") for value in raw_spacings
    ]
    if min(multipliers) <= 0 or min(spacings) <= 0:
        raise ValueError("Locked convergence sweep window must be positive")
    multiplier = max(multipliers)
    spacing = min(spacings)
    reference_multiplier = _finite_float(
        protocol.get("kpoint_reference_cutoff_multiplier"),
        "kpoint_reference_cutoff_multiplier",
    )
    reference_spacing = _finite_float(
        protocol.get("cutoff_reference_kpoint_spacing_inv_angstrom"),
        "cutoff_reference_kpoint_spacing_inv_angstrom",
    )
    if not math.isclose(multiplier, reference_multiplier, rel_tol=1e-12, abs_tol=1e-10):
        raise ValueError("Locked cutoff reference is not the strictest tested level")
    if not math.isclose(spacing, reference_spacing, rel_tol=1e-12, abs_tol=1e-10):
        raise ValueError("Locked k-point reference is not the densest tested level")
    base_wfc = _finite_float(base_ecutwfc_ry, "base_ecutwfc_ry")
    base_rho = _finite_float(base_ecutrho_ry, "base_ecutrho_ry")
    if base_wfc <= 0 or base_rho <= 0:
        raise ValueError("Locked base cutoffs must be positive")
    return {
        "cutoff_pair_multiplier": multiplier,
        "ecutwfc_ry": base_wfc * multiplier,
        "ecutrho_ry": base_rho * multiplier,
        "kpoint_spacing_inv_angstrom": spacing,
    }


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Provisional evidence {label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Provisional evidence {label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"Provisional evidence {label} must be finite")
    return result


def _integer(value: Any, label: str) -> int:
    result = _finite_float(value, label)
    rounded = int(round(result))
    if not math.isclose(result, rounded, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Provisional evidence {label} must be an integer")
    return rounded


def _boolean(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Provisional evidence {label} must be a canonical boolean")


def _json_value(value: Any, label: str) -> Any:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Provisional evidence {label} is invalid JSON") from exc
    try:
        # Round-trip also rejects values that cannot be represented canonically.
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Provisional evidence {label} is not canonical JSON") from exc


def _same_float(left: Any, right: Any, label: str) -> float:
    first = _finite_float(left, label)
    second = _finite_float(right, label)
    if not math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-10):
        raise ValueError(f"Provisional JSON/CSV mismatch: {label}")
    return first


def _compare_field(field: str, json_value: Any, csv_value: Any, point_id: str) -> None:
    label = f"{point_id}:{field}"
    if field in _FLOAT_FIELDS:
        _same_float(json_value, csv_value, label)
    elif field in _INT_FIELDS:
        if _integer(json_value, label) != _integer(csv_value, label):
            raise ValueError(f"Provisional JSON/CSV mismatch: {label}")
    elif field in _BOOL_FIELDS:
        if _boolean(json_value, label) != _boolean(csv_value, label):
            raise ValueError(f"Provisional JSON/CSV mismatch: {label}")
    elif field in _JSON_FIELDS or isinstance(json_value, (dict, list)):
        if _json_value(json_value, label) != _json_value(csv_value, label):
            raise ValueError(f"Provisional JSON/CSV mismatch: {label}")
    elif str(json_value if json_value is not None else "") != str(
        csv_value if csv_value is not None else ""
    ):
        raise ValueError(f"Provisional JSON/CSV mismatch: {label}")


def _components(value: Any, *, label: str, row_count: int) -> list[list[float]]:
    parsed = _json_value(value, label)
    if not isinstance(parsed, list) or len(parsed) != row_count:
        raise ValueError(f"Provisional evidence {label} has the wrong row count")
    result: list[list[float]] = []
    for index, vector in enumerate(parsed):
        if not isinstance(vector, list) or len(vector) != 3:
            raise ValueError(f"Provisional evidence {label}[{index}] is not a vector")
        result.append([
            _finite_float(component, f"{label}[{index}]") for component in vector
        ])
    return result


def _max_component_delta(
    first: list[list[float]], second: list[list[float]], *, label: str,
) -> float:
    if len(first) != len(second):
        raise ValueError(f"Provisional evidence {label} has incompatible shapes")
    deltas = [
        abs(left - right)
        for first_row, second_row in zip(first, second)
        for left, right in zip(first_row, second_row)
    ]
    if not deltas:
        raise ValueError(f"Provisional evidence {label} is empty")
    return max(deltas)


def _select_stable_tail(rows: list[dict[str, Any]], min_tail: int) -> dict[str, Any] | None:
    for index, row in enumerate(rows):
        tail = rows[index:]
        if len(tail) >= min_tail and all(
            _boolean(item["point_within_tolerances"], "point_within_tolerances")
            for item in tail
        ):
            return row
    return None


def _same_selection_number(left: Any, right: Any, label: str) -> None:
    _same_float(left, right, label)


def verify_provisional_sweep_evidence(
    *,
    summary: dict[str, Any],
    csv_rows: list[dict[str, str]],
    locked_study_points: Iterable[dict[str, Any]] | None = None,
    representative_ids: Iterable[str] | None = None,
    expected_protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify JSON/CSV equality and independently derive the sweep selection.

    Returns canonical point/anchor inventories and the re-derived global
    selection.  The function assumes an eligible provisional summary; any
    missing, non-finite, inconsistent, or scientifically non-passing value is
    rejected.
    """

    points = summary.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("Provisional sweep summary has no point evidence")
    if summary.get("point_count") != len(points) or len(csv_rows) != len(points):
        raise ValueError("Provisional sweep point counts differ")

    point_by_id: dict[str, dict[str, Any]] = {}
    for raw in points:
        if not isinstance(raw, dict):
            raise ValueError("Provisional sweep point evidence is not an object")
        point_id = str(raw.get("point_id") or "")
        if not point_id or point_id in point_by_id:
            raise ValueError("Provisional sweep point IDs are missing or duplicated")
        point_by_id[point_id] = raw
    csv_by_id = {str(row.get("point_id") or ""): row for row in csv_rows}
    if (
        len(csv_by_id) != len(csv_rows)
        or "" in csv_by_id
        or set(csv_by_id) != set(point_by_id)
    ):
        raise ValueError("Provisional JSON/CSV point inventories differ")

    for point_id, point in point_by_id.items():
        csv_row = csv_by_id[point_id]
        if set(csv_row) != set(point):
            missing = sorted(set(point) - set(csv_row))
            extra = sorted(set(csv_row) - set(point))
            raise ValueError(
                f"Provisional JSON/CSV columns differ for {point_id}: "
                f"missing={missing}, extra={extra}"
            )
        for field, value in point.items():
            _compare_field(field, value, csv_row[field], point_id)

    if locked_study_points is not None:
        locked = list(locked_study_points)
        locked_by_id = {str(point.get("point_id") or ""): point for point in locked}
        if (
            len(locked_by_id) != len(locked)
            or "" in locked_by_id
            or set(locked_by_id) != set(point_by_id)
        ):
            raise ValueError("Provisional evidence differs from locked sweep inventory")
        locked_fields = (
            "representative_id", "sweep_axis", "cutoff_level_index",
            "kpoint_level_index", "cutoff_pair_multiplier",
            "requested_kpoint_spacing_inv_angstrom", "ecutwfc_ry", "ecutrho_ry",
        )
        for point_id, locked_point in locked_by_id.items():
            point = point_by_id[point_id]
            for field in locked_fields:
                _compare_field(field, point.get(field), locked_point.get(field), point_id)
            expected_grid = "x".join(str(value) for value in locked_point["kpoints_grid"])
            if str(point.get("kpoints_grid") or "") != expected_grid:
                raise ValueError(f"Provisional point k-grid differs from sweep: {point_id}")

    protocol = summary.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Provisional convergence protocol is missing")
    if expected_protocol is not None and protocol != expected_protocol:
        raise ValueError("Provisional convergence protocol differs from locked protocol")
    energy_tol = _finite_float(
        protocol.get("energy_tolerance_mev_per_atom"), "energy tolerance"
    )
    force_tol = _finite_float(
        protocol.get("force_component_tolerance_ev_per_angstrom"), "force tolerance"
    )
    stress_tol = _finite_float(
        protocol.get("stress_component_tolerance_kbar"), "stress tolerance"
    )
    min_tail = _integer(protocol.get("min_stable_tail_points"), "minimum stable tail")
    if min(energy_tol, force_tol, stress_tol) <= 0 or min_tail < 2:
        raise ValueError("Provisional convergence protocol is unsafe")

    by_rep: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        rep_id = str(point.get("representative_id") or "")
        if not rep_id:
            raise ValueError("Provisional point has no representative ID")
        by_rep[rep_id].append(point)
        if point.get("convergence_gate_status") != "converged_output":
            raise ValueError(f"Provisional point did not converge: {point['point_id']}")
        if _json_value(point.get("gate_failures"), "gate_failures") != []:
            raise ValueError(f"Provisional point has gate failures: {point['point_id']}")

    expected_reps = set(str(value) for value in representative_ids or by_rep)
    if not expected_reps or set(by_rep) != expected_reps:
        raise ValueError("Provisional representative inventory is incomplete")
    if summary.get("candidate_count") != len(expected_reps):
        raise ValueError("Provisional representative count is inconsistent")
    expected_counts = dict(Counter("converged_output" for _ in points))
    if summary.get("gate_status_counts") != expected_counts:
        raise ValueError("Provisional gate-status count is inconsistent")

    selections = summary.get("selections_by_representative")
    if not isinstance(selections, dict) or set(selections) != expected_reps:
        raise ValueError("Provisional selections do not cover every representative")

    derived_selections: dict[str, dict[str, Any]] = {}
    anchors_by_rep: dict[str, dict[str, Any]] = {}
    true_stable_tail_ids: set[str] = set()
    for representative_id, rep_points in sorted(by_rep.items()):
        anchors = [point for point in rep_points if point.get("sweep_axis") == "anchor"]
        if len(anchors) != 1:
            raise ValueError(f"Provisional anchor count is invalid: {representative_id}")
        anchor = anchors[0]
        anchors_by_rep[representative_id] = anchor
        atom_count = _integer(anchor.get("num_atoms"), f"{representative_id}:num_atoms")
        if atom_count < 1:
            raise ValueError(f"Provisional atom count is invalid: {representative_id}")
        anchor_energy = _finite_float(
            anchor.get("energy_ev_per_atom"), f"{representative_id}:anchor energy"
        )
        anchor_forces = _components(
            anchor.get("force_components_ev_per_angstrom"),
            label=f"{representative_id}:anchor forces", row_count=atom_count,
        )
        anchor_stress = _components(
            anchor.get("stress_components_kbar"),
            label=f"{representative_id}:anchor stress", row_count=3,
        )
        for point in rep_points:
            point_id = point["point_id"]
            if _integer(point.get("num_atoms"), f"{point_id}:num_atoms") != atom_count:
                raise ValueError(f"Provisional atom count changed within {representative_id}")
            forces = _components(
                point.get("force_components_ev_per_angstrom"),
                label=f"{point_id}:forces", row_count=atom_count,
            )
            stress = _components(
                point.get("stress_components_kbar"),
                label=f"{point_id}:stress", row_count=3,
            )
            expected_energy = (
                _finite_float(point.get("total_energy_ry"), f"{point_id}:total energy")
                * RY_TO_EV
                / atom_count
            )
            _same_float(
                point.get("energy_ev_per_atom"), expected_energy,
                f"{point_id}:energy_ev_per_atom_from_total_energy_ry",
            )
            expected_energy_delta = abs(
                _finite_float(point.get("energy_ev_per_atom"), f"{point_id}:energy")
                - anchor_energy
            ) * 1000.0
            expected_force_delta = _max_component_delta(
                forces, anchor_forces, label=f"{point_id}:forces"
            )
            expected_stress_delta = _max_component_delta(
                stress, anchor_stress, label=f"{point_id}:stress"
            )
            recorded = (
                ("energy_delta_to_anchor_mev_per_atom", expected_energy_delta),
                ("max_force_delta_to_anchor_ev_per_angstrom", expected_force_delta),
                ("max_stress_delta_to_anchor_kbar", expected_stress_delta),
            )
            for field, expected in recorded:
                _same_float(point.get(field), expected, f"{point_id}:{field}")
            expected_within = (
                expected_energy_delta <= energy_tol
                and expected_force_delta <= force_tol
                and expected_stress_delta <= stress_tol
            )
            if _boolean(point.get("point_within_tolerances"), point_id) != expected_within:
                raise ValueError(f"Provisional point pass flag is inconsistent: {point_id}")

        cutoff_rows = sorted(
            [point for point in rep_points if point.get("sweep_axis") in {"cutoff_pair", "anchor"}],
            key=lambda point: _integer(point.get("cutoff_level_index"), "cutoff level"),
        )
        kpoint_rows = sorted(
            [point for point in rep_points if point.get("sweep_axis") in {"kpoint", "anchor"}],
            key=lambda point: _integer(point.get("kpoint_level_index"), "k-point level"),
        )
        if len({point["cutoff_level_index"] for point in cutoff_rows}) != len(cutoff_rows):
            raise ValueError(f"Duplicate cutoff levels: {representative_id}")
        if len({point["kpoint_level_index"] for point in kpoint_rows}) != len(kpoint_rows):
            raise ValueError(f"Duplicate k-point levels: {representative_id}")
        cutoff = _select_stable_tail(cutoff_rows, min_tail)
        kpoint = _select_stable_tail(kpoint_rows, min_tail)
        if cutoff is None or kpoint is None:
            raise ValueError(f"Provisional stable tail is missing: {representative_id}")
        true_stable_tail_ids.update(
            point["point_id"] for point in cutoff_rows[cutoff_rows.index(cutoff):]
        )
        true_stable_tail_ids.update(
            point["point_id"] for point in kpoint_rows[kpoint_rows.index(kpoint):]
        )
        derived = {
            "selected_cutoff_pair": {
                "multiplier": _finite_float(cutoff["cutoff_pair_multiplier"], "multiplier"),
                "ecutwfc_ry": _finite_float(cutoff["ecutwfc_ry"], "ecutwfc"),
                "ecutrho_ry": _finite_float(cutoff["ecutrho_ry"], "ecutrho"),
                "point_id": cutoff["point_id"],
            },
            "selected_kpoint": {
                "spacing_inv_angstrom": _finite_float(
                    kpoint["requested_kpoint_spacing_inv_angstrom"], "k-point spacing"
                ),
                "kpoints_grid": kpoint["kpoints_grid"],
                "point_id": kpoint["point_id"],
            },
            "anchor_point_id": anchor["point_id"],
        }
        reported = selections[representative_id]
        if not isinstance(reported, dict):
            raise ValueError(f"Provisional selection is invalid: {representative_id}")
        if reported.get("anchor_point_id") != derived["anchor_point_id"]:
            raise ValueError(f"Provisional anchor selection is inconsistent: {representative_id}")
        for section, numeric_fields in (
            ("selected_cutoff_pair", ("multiplier", "ecutwfc_ry", "ecutrho_ry")),
            ("selected_kpoint", ("spacing_inv_angstrom",)),
        ):
            actual_section = reported.get(section)
            expected_section = derived[section]
            if not isinstance(actual_section, dict):
                raise ValueError(f"Provisional selection is incomplete: {representative_id}")
            for field in numeric_fields:
                _same_selection_number(
                    actual_section.get(field), expected_section[field],
                    f"{representative_id}:{section}.{field}",
                )
            for field in set(expected_section) - set(numeric_fields):
                if actual_section.get(field) != expected_section[field]:
                    raise ValueError(
                        f"Provisional selection is inconsistent: {representative_id}:{section}.{field}"
                    )
        derived_selections[representative_id] = derived

    for point in points:
        expected = point["point_id"] in true_stable_tail_ids
        if _boolean(point.get("stable_tail_from_this_level"), point["point_id"]) != expected:
            raise ValueError(
                f"Provisional stable-tail flag is inconsistent: {point['point_id']}"
            )

    cutoff_values = [value["selected_cutoff_pair"] for value in derived_selections.values()]
    kpoint_values = [value["selected_kpoint"] for value in derived_selections.values()]
    derived_global = {
        "ecutwfc_ry": max(value["ecutwfc_ry"] for value in cutoff_values),
        "ecutrho_ry": max(value["ecutrho_ry"] for value in cutoff_values),
        "cutoff_pair_multiplier": max(value["multiplier"] for value in cutoff_values),
        "kpoint_spacing_inv_angstrom": min(
            value["spacing_inv_angstrom"] for value in kpoint_values
        ),
    }
    reported_global = summary.get("provisional_global_selection")
    if not isinstance(reported_global, dict) or set(reported_global) != set(derived_global):
        raise ValueError("Provisional global selection is invalid")
    for field, expected in derived_global.items():
        _same_float(reported_global.get(field), expected, f"global selection:{field}")

    return {
        "point_by_id": point_by_id,
        "anchors_by_representative": anchors_by_rep,
        "selections_by_representative": derived_selections,
        "global_selection": derived_global,
    }
