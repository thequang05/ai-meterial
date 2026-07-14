"""Reconstruct unrelaxed candidate CIFs from Phase 2 site substitutions.

The GraphVAE does not generate lattice vectors or fractional coordinates.  A
structural candidate is therefore a *site-substituted prototype*: it retains
the prototype lattice and coordinates and changes only the decoded species.
Every reconstruction must prove that the graph node order and the cached
prototype Structure order agree before applying substitutions.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

from pymatgen.core import Composition, Element, Structure
from pymatgen.io.cif import CifWriter


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_STRUCTURE_CACHE = _PROJECT_ROOT / "research" / "phase_3" / "cache" / "structures.sqlite"


class StructureBuildError(ValueError):
    """A candidate cannot be safely reconstructed as a structure."""


def _cache_material_id(uid: str) -> str:
    """Convert Neo4j's ``MP_mp-...`` UID to raw-cache ``mp-...`` UID."""
    return uid[3:] if uid.startswith("MP_") else uid


def _graph_atomic_numbers(graph: Any) -> list[int]:
    if not hasattr(graph, "x"):
        raise StructureBuildError("graph_missing_node_features")
    return [int(z) for z in graph.x.view(-1).detach().cpu().tolist()]


def load_prototype_structure(
    uid: str,
    structure_cache: Path = DEFAULT_STRUCTURE_CACHE,
) -> Structure:
    """Load a prototype structure from the Phase 3 sqlite CIF cache."""
    structure_cache = Path(structure_cache)
    if not structure_cache.exists():
        raise FileNotFoundError(
            f"Structure cache is missing: {structure_cache}. "
            "Build it from the Materials Project raw JSON before exporting CIFs."
        )

    with sqlite3.connect(structure_cache) as conn:
        row = conn.execute(
            "SELECT cif FROM structures WHERE material_id = ?",
            (_cache_material_id(uid),),
        ).fetchone()
    if row is None or not row[0]:
        raise StructureBuildError(f"prototype_structure_not_found:{uid}")
    try:
        return Structure.from_str(row[0], fmt="cif")
    except Exception as exc:  # noqa: BLE001
        raise StructureBuildError(f"prototype_cif_parse_failed:{uid}:{exc}") from exc


def verify_graph_structure_alignment(graph: Any, structure: Structure) -> None:
    """Fail unless the graph and Structure have the identical site ordering."""
    graph_numbers = _graph_atomic_numbers(graph)
    structure_numbers = [int(z) for z in structure.atomic_numbers]
    if len(graph_numbers) != len(structure_numbers):
        raise StructureBuildError(
            "graph_structure_order_mismatch:"
            f"node_count graph={len(graph_numbers)} structure={len(structure_numbers)}"
        )
    if graph_numbers != structure_numbers:
        raise StructureBuildError("graph_structure_order_mismatch:atomic_numbers")


def apply_site_substitutions(
    structure: Structure,
    original_atomic_numbers: Iterable[int],
    candidate_atomic_numbers: Iterable[int],
) -> tuple[Structure, list[dict[str, int | str]]]:
    """Copy ``structure`` and apply ordered atomic-number substitutions.

    The input structure is never mutated.  Coordinates and lattice are copied
    unchanged, hence the returned structure is explicitly *unrelaxed*.
    """
    original = [int(z) for z in original_atomic_numbers]
    candidate = [int(z) for z in candidate_atomic_numbers]
    structure_numbers = [int(z) for z in structure.atomic_numbers]
    if original != structure_numbers:
        raise StructureBuildError("graph_structure_order_mismatch:atomic_numbers")
    if len(candidate) != len(original):
        raise StructureBuildError(
            f"candidate_node_count_mismatch:expected={len(original)} got={len(candidate)}"
        )

    reconstructed = structure.copy()
    substitutions: list[dict[str, int | str]] = []
    for site_index, (from_z, to_z) in enumerate(zip(original, candidate)):
        if from_z == to_z:
            continue
        try:
            from_symbol = Element.from_Z(from_z).symbol
            to_symbol = Element.from_Z(to_z).symbol
        except Exception as exc:  # noqa: BLE001
            raise StructureBuildError(f"invalid_atomic_number_at_site:{site_index}") from exc
        reconstructed.replace(site_index, Element.from_Z(to_z))
        substitutions.append({
            "site_index": site_index,
            "from_Z": from_z,
            "to_Z": to_z,
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
        })
    return reconstructed, substitutions


def candidate_atomic_numbers_from_substitutions(
    original_atomic_numbers: Iterable[int],
    substitutions: Iterable[dict[str, Any]],
) -> list[int]:
    """Reconstruct candidate site species and verify recorded provenance."""
    candidate = [int(z) for z in original_atomic_numbers]
    used_sites: set[int] = set()
    for substitution in substitutions:
        try:
            site = int(substitution["site_index"])
            from_z = int(substitution["from_Z"])
            to_z = int(substitution["to_Z"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StructureBuildError(f"invalid_substitution_record:{substitution}") from exc
        if site < 0 or site >= len(candidate):
            raise StructureBuildError(f"substitution_site_out_of_range:{site}")
        if site in used_sites:
            raise StructureBuildError(f"duplicate_substitution_site:{site}")
        if candidate[site] != from_z:
            raise StructureBuildError(
                f"substitution_source_mismatch:site={site}:expected={candidate[site]}:recorded={from_z}"
            )
        candidate[site] = to_z
        used_sites.add(site)
    return candidate


def write_candidate_cif(structure: Structure, output_path: Path) -> None:
    """Write an unrelaxed candidate CIF."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # A substituted site can inherit a prototype label that collides with an
    # existing site of the new species (for example Ti2 -> Ta while Ta1 is
    # already present). CIF requires unique atom-site labels.
    clean = structure.copy().relabel_sites()
    CifWriter(clean).write_file(str(output_path))


def build_candidate_cif(
    *,
    candidate_id: str,
    prototype_uid: str,
    graph: Any,
    substitutions: Iterable[dict[str, Any]],
    expected_formula: str,
    output_path: Path,
    structure_cache: Path = DEFAULT_STRUCTURE_CACHE,
) -> dict[str, Any]:
    """Build one CIF and return auditable structural provenance metadata."""
    prototype = load_prototype_structure(prototype_uid, structure_cache)
    verify_graph_structure_alignment(graph, prototype)
    original_numbers = _graph_atomic_numbers(graph)
    candidate_numbers = candidate_atomic_numbers_from_substitutions(
        original_numbers, substitutions
    )
    candidate, applied = apply_site_substitutions(
        prototype, original_numbers, candidate_numbers
    )

    actual_formula = candidate.composition.reduced_formula
    try:
        expected_reduced = Composition(expected_formula).reduced_formula
    except (ValueError, TypeError) as exc:
        raise StructureBuildError(f"invalid_manifest_formula:{expected_formula}") from exc
    if actual_formula != expected_reduced:
        raise StructureBuildError(
            f"candidate_formula_mismatch:expected={expected_reduced}:actual={actual_formula}"
        )

    write_candidate_cif(candidate, output_path)
    # Read back immediately: an output path is not a valid candidate artifact
    # until pymatgen can parse it and its composition still agrees.
    parsed = Structure.from_file(output_path)
    if parsed.composition.reduced_formula != expected_reduced:
        raise StructureBuildError("written_cif_formula_mismatch")

    return {
        "candidate_id": candidate_id,
        "prototype_uid": prototype_uid,
        "candidate_formula": expected_reduced,
        "substitutions_json": applied,
        "num_substitutions": len(applied),
        "cif_path": str(output_path),
        "structure_status": "unrelaxed",
    }
