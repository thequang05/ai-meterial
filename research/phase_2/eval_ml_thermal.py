"""Machine-learning-only thermal evaluation for refractory-carbide candidates.

This script is a DFT-free evaluation pipeline for generated candidate materials.
It is intended for environments where Quantum ESPRESSO is not feasible (for
example, a laptop with limited disk space, no HPC cluster, and tight wall-clock
budgets). It produces auditable, content-hash-bound output that records every
input that fed into the surrogate so an operator can later swap a DFT check
in for the candidates the surrogate ranks highest.

What it does (no DFT required):

  1. Read a candidate manifest (``research/phase_2/reports/
     w_c_structural_validation_v1.json``) plus the CHGNet-relaxed CIFs.
  2. Build a PyG graph for each CHGNet-relaxed CIF and score it with the
     audited formation-energy GNN.
  3. Compare the predicted formation energy against the lowest-energy MP
     reference in the same chemical system (subsystem). The energy gap is
     reported as ``ml_estimated_hull_gap_ev_per_atom``.
  4. Cross-check the candidate formula against Materials Project: is the
     reduced composition already known? If so, how does the MP value compare?
  5. Apply thermal-proxy heuristics (high-temperature refractory-carbide
     chemistry rules, formation-energy window, forbidden elements) to assign
     a thermal-friendly flag.
  6. Rank candidates, write a CSV manifest and a JSON summary with the
     SHA-256 of every input that fed into the evaluation.

What it explicitly does NOT do:

  * Run any Quantum ESPRESSO calculation.
  * Touch ``/private/tmp/qe_scratch`` or the convergence sweep queue.
  * Produce a convex hull number that can be reported as a DFT-validated
    formation-energy-above-hull. The ML estimate is labelled
    ``ml_estimated_hull_gap_ev_per_atom`` so it can never be confused with
    the audited DFT value.

The script never invokes external network calls, never launches MPI, and
never downloads data. It only reads the local GraphVAE pipeline artifacts
and writes new CSV/JSON artifacts in the requested output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pymatgen.core import Composition, Structure
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT / "research" / "phase_2") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research" / "phase_2"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "phase_2" / "generation"))

from gnn_model import MaterialGNN  # noqa: E402

WORKFLOW_VERSION = "ml_only_thermal_eval_v1"

REFACTORY_ELEMENTS = frozenset({"Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"})
CERAMIC_FORMERS = frozenset({"B", "C", "N", "O", "Al", "Si"})
FORBIDDEN_ELEMENTS = frozenset({"H", "Li", "F", "Na", "Cl", "K", "Br", "Rb", "I", "Cs"})

# Thermal-friendly window in eV/atom. This is intentionally generous: the
# surrogate is uncertain on the order of 0.1 eV/atom, so anything clearly
# more negative than -0.25 eV/atom is treated as a candidate worth pursuing
# further. The DFT-audited threshold (0.025 eV/atom above hull) is much
# tighter and cannot be reproduced by an ML-only surrogate.
THERMAL_FORMATION_ENERGY_UPPER_BOUND_EV_PER_ATOM = -0.25
THERMAL_FORMATION_ENERGY_DESIRABLE_BOUND_EV_PER_ATOM = -0.45

# When the surrogate disagrees with the lowest-energy MP reference in the
# same subsystem by less than this many eV/atom, treat the candidate as
# potentially competitive with the best known phase. Beyond this, it is
# likely too high in energy to be synthesizable.
MP_REFERENCE_DISCREPANCY_EV_PER_ATOM = 0.5

DEFAULT_VALIDATION_REPORT = (
    PROJECT_ROOT / "research" / "phase_2" / "reports" / "w_c_structural_validation_v1.json"
)
DEFAULT_GNN_CHECKPOINT = PROJECT_ROOT / "research" / "phase_2" / "models" / "gnn_formation_energy_grouped_v1.pt"
DEFAULT_MATERIALS_CSV = PROJECT_ROOT / "research" / "phase_2" / "data" / "processed" / "materials.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _chemsys(elements: set[str]) -> str:
    return "-".join(sorted(elements))


def _formula_signature(formula: str) -> str:
    return Composition(formula).reduced_formula


def _load_struct(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_numbers(structure: Structure) -> torch.Tensor:
    return torch.tensor(
        [int(site.specie.Z) for site in structure], dtype=torch.long
    ).view(-1, 1)


def _edges_within_cutoff(structure: Structure, cutoff: float = 4.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a simple k-NN-style edge list from periodic neighbor search.

    The graph VAE uses a learned edge decoder, but for evaluation we only need
    a graph that the GNN can ingest. A short periodic neighbor cutoff is
    sufficient and avoids depending on the original generation edge_attr.
    """
    src_list: list[int] = []
    dst_list: list[int] = []
    edge_attrs: list[float] = []
    sites = list(structure.sites)
    for i, site_i in enumerate(sites):
        neighbors = structure.get_neighbors(site_i, cutoff)
        for neighbor in neighbors:
            j = int(neighbor.index) if hasattr(neighbor, "index") else int(neighbor.site_index)
            if i == j:
                continue
            src_list.append(i)
            dst_list.append(j)
            edge_attrs.append(float(neighbor.nn_distance))
    if not src_list:
        # No neighbors within cutoff — emit a self-loop so the GNN has at least
        # one edge to operate on. The numeric prediction will be unreliable;
        # we surface that case as ``graph_empty`` in the report.
        src_list = [0]
        dst_list = [0]
        edge_attrs = [0.0]
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 1)
    return edge_index, edge_attr


def _build_pyg_data(structure: Structure, uid: str) -> Data:
    edge_index, edge_attr = _edges_within_cutoff(structure)
    data = Data(
        x=_atomic_numbers(structure),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    data.material_uid = uid
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    return data


def _chemistry_flags(formula: str) -> dict[str, Any]:
    composition = Composition(formula)
    elements = {element.symbol for element in composition.elements}
    refractory_present = bool(elements & REFACTORY_ELEMENTS)
    ceramic_present = bool(elements & CERAMIC_FORMERS)
    forbidden_present = bool(elements & FORBIDDEN_ELEMENTS)
    return {
        "elements": sorted(elements),
        "refractory_present": refractory_present,
        "ceramic_present": ceramic_present,
        "forbidden_present": forbidden_present,
        "passes_thermal_chemistry": (
            refractory_present
            and ceramic_present
            and not forbidden_present
        ),
    }


def _load_mp_reference(materials_csv: Path) -> dict[str, list[dict[str, Any]]]:
    """Group MP materials by their chemical subsystem for cheap lookup."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with materials_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                composition = Composition(row["formula"])
            except Exception:
                continue
            try:
                energy = float(row["formation_energy_per_atom"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(energy):
                continue
            chemsys = _chemsys({element.symbol for element in composition.elements})
            grouped[chemsys].append(
                {
                    "uid": row.get("uid") or row.get("material_id") or "",
                    "formula": composition.reduced_formula,
                    "formation_energy_per_atom": energy,
                    "chemsys": chemsys,
                }
            )
    return grouped


def _best_mp_reference(
    mp_grouped: dict[str, list[dict[str, Any]]],
    formula: str,
) -> dict[str, Any]:
    """Find the lowest-energy MP reference in the same chemical subsystem.

    Falls back to smaller subsystems if the full subsystem has no entries.
    This matches the spirit of a DFT-audited hull: if a quaternary candidate
    has no direct quaternary competition, the best reference in any binary
    or ternary subsystem is still informative. Surrogate accuracy is lower
    in this fallback mode, so the status is reported separately.
    """
    composition = Composition(formula)
    elements = {element.symbol for element in composition.elements}
    full_chemsys = _chemsys(elements)

    candidate_keys: list[str] = []
    for size in range(len(elements), 0, -1):
        from itertools import combinations
        for combo in combinations(sorted(elements), size):
            candidate_keys.append(_chemsys(set(combo)))

    seen: set[str] = set()
    for chemsys in candidate_keys:
        if chemsys in seen:
            continue
        seen.add(chemsys)
        entries = mp_grouped.get(chemsys, [])
        if not entries:
            continue
        best = min(entries, key=lambda entry: entry["formation_energy_per_atom"])
        return {
            "chemsys": chemsys,
            "best_uid": best["uid"],
            "best_formula": best["formula"],
            "best_formation_energy_per_atom": best["formation_energy_per_atom"],
            "best_from_same_composition": (
                _formula_signature(best["formula"])
                == _formula_signature(formula)
            ),
            "entry_count": len(entries),
            "is_fallback_subsystem": chemsys != full_chemsys,
            "candidate_full_chemsys": full_chemsys,
        }
    return None


def _select_cif_path(
    candidate_id: str,
    primary_dir: Path,
    fallback_dir: Path | None = None,
) -> Path | None:
    """Find the CHGNet-relaxed CIF for a candidate by id.

    Searches ``primary_dir`` first, then ``fallback_dir``. Matches any CIF
    whose filename contains the candidate id. The fallback covers the case
    where the structural-validation CIFs exist but the convergence-source
    bootstrap queue was prepared for only a subset of the candidates.
    """
    search_dirs = [primary_dir]
    if fallback_dir is not None:
        search_dirs.append(fallback_dir)
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for cif_path in sorted(search_dir.glob("*_chgnet_relaxed.cif")):
            if candidate_id in cif_path.name:
                return cif_path
        for cif_path in sorted(search_dir.glob("input_chgnet_relaxed.cif")):
            if candidate_id in cif_path.parent.name:
                return cif_path
    return None


def evaluate_candidate(
    candidate: dict[str, Any],
    *,
    cif_path: Path | None,
    gnn: MaterialGNN,
    device: torch.device,
    mp_grouped: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Score one candidate with the GNN and compare against MP references."""
    formula = candidate["formula"]
    chemistry = _chemistry_flags(formula)
    mp_ref = _best_mp_reference(mp_grouped, formula)

    record: dict[str, Any] = {
        "rank": candidate.get("rank"),
        "candidate_id": candidate["candidate_id"],
        "formula": formula,
        "num_sites": candidate.get("num_sites"),
        "gnn_formation_energy_pre_chgnet_ev_per_atom": candidate.get(
            "gnn_formation_energy_ev_per_atom"
        ),
        "chgnet_relaxation_status": candidate.get("chgnet_relaxation_status"),
        "chgnet_energy_change_ev_per_atom": candidate.get(
            "chgnet_energy_change_ev_per_atom"
        ),
        "chemistry": chemistry,
        "ml_estimated_hull_gap_ev_per_atom": None,
        "mp_reference": mp_ref,
        "mp_relative_gap_ev_per_atom": None,
        "cif_path": str(cif_path) if cif_path else "",
        "cif_sha256": _sha256(cif_path) if cif_path else "",
        "graph_empty": False,
        "thermal_proxy_status": "not_evaluated",
        "thermal_proxy_reason": "",
    }

    if cif_path is None or not cif_path.exists():
        record["thermal_proxy_status"] = "missing_cif"
        record["thermal_proxy_reason"] = (
            "CHGNet-relaxed CIF not found under convergence_source_v1/"
        )
        return record

    try:
        structure = Structure.from_file(str(cif_path))
    except Exception as exc:  # noqa: BLE001
        record["thermal_proxy_status"] = "cif_unreadable"
        record["thermal_proxy_reason"] = f"{type(exc).__name__}: {exc}"
        return record

    if len(structure) == 0:
        record["thermal_proxy_status"] = "empty_structure"
        record["thermal_proxy_reason"] = "CIF parsed with zero sites"
        return record

    actual_signature = _formula_signature(structure.composition.reduced_formula)
    expected_signature = _formula_signature(formula)
    if actual_signature != expected_signature:
        record["thermal_proxy_status"] = "formula_mismatch"
        record["thermal_proxy_reason"] = (
            f"expected={expected_signature}, cif={actual_signature}"
        )
        return record

    graph = _build_pyg_data(structure, uid=candidate["candidate_id"])
    if graph.edge_index.size(1) <= 1:
        record["graph_empty"] = True

    graph = graph.to(device)
    with torch.no_grad():
        gnn_pred = float(gnn(graph).detach().cpu().reshape(-1)[0])

    record["gnn_formation_energy_post_chgnet_ev_per_atom"] = round(gnn_pred, 6)
    record["gnn_delta_from_chgnet_ev_per_atom"] = round(
        gnn_pred - float(candidate.get("gnn_formation_energy_ev_per_atom") or 0.0),
        6,
    )

    if mp_ref is not None:
        gap = gnn_pred - float(mp_ref["best_formation_energy_per_atom"])
        record["ml_estimated_hull_gap_ev_per_atom"] = round(gap, 6)
        record["mp_relative_gap_ev_per_atom"] = round(gap, 6)
        if gap <= MP_REFERENCE_DISCREPANCY_EV_PER_ATOM:
            record["thermal_proxy_status"] = "competitive_with_best_mp"
            record["thermal_proxy_reason"] = (
                f"surrogate within {MP_REFERENCE_DISCREPANCY_EV_PER_ATOM:.2f} eV/atom "
                f"of best MP reference in subsystem"
            )
        else:
            record["thermal_proxy_status"] = "above_mp_best"
            record["thermal_proxy_reason"] = (
                f"surrogate {gap:.3f} eV/atom above best MP reference"
            )
    else:
        record["thermal_proxy_status"] = "no_mp_reference"
        record["thermal_proxy_reason"] = (
            "No MP entries found in the same chemical subsystem"
        )

    if not chemistry["passes_thermal_chemistry"]:
        record["thermal_proxy_status"] = "chemistry_rejected"
        record["thermal_proxy_reason"] = (
            "Chemistry does not satisfy refractory+ceramic, no forbidden"
        )
        return record

    if gnn_pred > THERMAL_FORMATION_ENERGY_UPPER_BOUND_EV_PER_ATOM:
        if record["thermal_proxy_status"] not in {"chemistry_rejected", "no_mp_reference"}:
            record["thermal_proxy_status"] = "energy_too_high"
            record["thermal_proxy_reason"] = (
                f"GNN formation energy {gnn_pred:.3f} > "
                f"{THERMAL_FORMATION_ENERGY_UPPER_BOUND_EV_PER_ATOM:.3f} eV/atom"
            )

    return record


def _rank_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order records so the most thermal-friendly candidate is rank 1.

    The surrogate ranking is heuristic. Priority order:
      1. chemistry_pass + competitive_with_best_mp + energy <= desirable bound
      2. chemistry_pass + competitive_with_best_mp
      3. chemistry_pass + low GNN formation energy
      4. chemistry_pass only
      5. everything else (failures first)
    """

    status_priority = {
        "competitive_with_best_mp": 0,
        "above_mp_best": 1,
        "no_mp_reference": 2,
        "energy_too_high": 3,
        "chemistry_rejected": 4,
        "missing_cif": 5,
        "cif_unreadable": 6,
        "empty_structure": 7,
        "formula_mismatch": 8,
        "graph_empty": 9,
        "not_evaluated": 10,
    }

    def sort_key(record: dict[str, Any]) -> tuple[int, float, float]:
        # Step 1: chemistry must pass first.
        chemistry_pass = bool(record["chemistry"]["passes_thermal_chemistry"])
        if not chemistry_pass:
            return (0, 0, 0)
        # Step 2: status priority among chemistry-pass candidates. We want
        # actually-evaluated candidates (CIF loaded, GNN ran) to outrank
        # missing/unreadable candidates.
        status = record["thermal_proxy_status"]
        evaluated_statuses = {
            "competitive_with_best_mp",
            "above_mp_best",
            "no_mp_reference",
            "energy_too_high",
        }
        if status in evaluated_statuses:
            evaluated_flag = 0
        elif status == "chemistry_rejected":
            return (1, 0, 0)
        else:
            return (2, 0, 0)
        status_priority_local = {
            "competitive_with_best_mp": 0,
            "no_mp_reference": 1,
            "above_mp_best": 2,
            "energy_too_high": 3,
        }
        status_first = status_priority_local.get(status, 99)
        mp_ref = record.get("mp_reference") or {}
        fallback_penalty = 1 if mp_ref.get("is_fallback_subsystem") else 0
        energy = record.get("gnn_formation_energy_post_chgnet_ev_per_atom")
        energy_value = float(energy) if isinstance(energy, (int, float)) else float("inf")
        return (evaluated_flag, fallback_penalty, status_first, energy_value)

    ranked = sorted(records, key=sort_key)
    for new_rank, record in enumerate(ranked, start=1):
        record["ml_rank"] = new_rank
    return ranked


def _write_outputs(
    records: list[dict[str, Any]],
    *,
    validation_report_path: Path,
    cif_dir: Path,
    materials_csv: Path,
    gnn_checkpoint: Path,
    output_dir: Path,
    audit_inputs: dict[str, str],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ml_thermal_evaluation.csv"
    json_path = output_dir / "ml_thermal_evaluation.json"

    fieldnames = [
        "ml_rank",
        "rank",
        "candidate_id",
        "formula",
        "num_sites",
        "chemistry_pass",
        "refractory_present",
        "ceramic_present",
        "forbidden_present",
        "gnn_formation_energy_pre_chgnet_ev_per_atom",
        "gnn_formation_energy_post_chgnet_ev_per_atom",
        "gnn_delta_from_chgnet_ev_per_atom",
        "chgnet_energy_change_ev_per_atom",
        "ml_estimated_hull_gap_ev_per_atom",
        "mp_relative_gap_ev_per_atom",
        "mp_best_formula",
        "mp_best_formation_energy_per_atom",
        "mp_best_uid",
        "mp_best_from_same_composition",
        "mp_subsystem",
        "mp_subsystem_entry_count",
        "mp_is_fallback_subsystem",
        "thermal_proxy_status",
        "thermal_proxy_reason",
        "graph_empty",
        "cif_path",
        "cif_sha256",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            chemistry = record["chemistry"]
            mp_ref = record["mp_reference"] or {}
            writer.writerow(
                {
                    "ml_rank": record["ml_rank"],
                    "rank": record["rank"],
                    "candidate_id": record["candidate_id"],
                    "formula": record["formula"],
                    "num_sites": record.get("num_sites"),
                    "chemistry_pass": chemistry["passes_thermal_chemistry"],
                    "refractory_present": chemistry["refractory_present"],
                    "ceramic_present": chemistry["ceramic_present"],
                    "forbidden_present": chemistry["forbidden_present"],
                    "gnn_formation_energy_pre_chgnet_ev_per_atom": (
                        record.get("gnn_formation_energy_pre_chgnet_ev_per_atom")
                    ),
                    "gnn_formation_energy_post_chgnet_ev_per_atom": (
                        record.get("gnn_formation_energy_post_chgnet_ev_per_atom")
                    ),
                    "gnn_delta_from_chgnet_ev_per_atom": (
                        record.get("gnn_delta_from_chgnet_ev_per_atom")
                    ),
                    "chgnet_energy_change_ev_per_atom": (
                        record.get("chgnet_energy_change_ev_per_atom")
                    ),
                    "ml_estimated_hull_gap_ev_per_atom": (
                        record.get("ml_estimated_hull_gap_ev_per_atom")
                    ),
                    "mp_relative_gap_ev_per_atom": (
                        record.get("mp_relative_gap_ev_per_atom")
                    ),
                    "mp_best_formula": mp_ref.get("best_formula", ""),
                    "mp_best_formation_energy_per_atom": mp_ref.get(
                        "best_formation_energy_per_atom"
                    ),
                    "mp_best_uid": mp_ref.get("best_uid", ""),
                    "mp_best_from_same_composition": mp_ref.get(
                        "best_from_same_composition"
                    ),
                    "mp_subsystem": mp_ref.get("chemsys", ""),
                    "mp_subsystem_entry_count": mp_ref.get("entry_count"),
                    "mp_is_fallback_subsystem": mp_ref.get(
                        "is_fallback_subsystem"
                    ),
                    "thermal_proxy_status": record["thermal_proxy_status"],
                    "thermal_proxy_reason": record["thermal_proxy_reason"],
                    "graph_empty": record["graph_empty"],
                    "cif_path": record["cif_path"],
                    "cif_sha256": record["cif_sha256"],
                }
            )

    status_counts: dict[str, int] = defaultdict(int)
    for record in records:
        status_counts[record["thermal_proxy_status"]] += 1
    chemistry_pass_count = sum(
        1 for record in records if record["chemistry"]["passes_thermal_chemistry"]
    )
    competitive_count = status_counts.get("competitive_with_best_mp", 0)

    summary = {
        "workflow_version": WORKFLOW_VERSION,
        "candidate_count": len(records),
        "chemistry_pass_count": chemistry_pass_count,
        "competitive_with_best_mp_count": competitive_count,
        "status_counts": dict(status_counts),
        "thermal_formation_energy_upper_bound_ev_per_atom": (
            THERMAL_FORMATION_ENERGY_UPPER_BOUND_EV_PER_ATOM
        ),
        "thermal_formation_energy_desirable_bound_ev_per_atom": (
            THERMAL_FORMATION_ENERGY_DESIRABLE_BOUND_EV_PER_ATOM
        ),
        "mp_reference_discrepancy_ev_per_atom": MP_REFERENCE_DISCREPANCY_EV_PER_ATOM,
        "audit_inputs": audit_inputs,
        "scientific_limit": (
            "Machine-learning surrogate only. Values are not Quantum ESPRESSO "
            "outputs, not audited formation-energy-above-hull, and not evidence "
            "of high-temperature performance. Run DFT on the top-ranked "
            "candidates when an HPC cluster is available."
        ),
    }
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=DEFAULT_VALIDATION_REPORT,
        help="Path to w_c_structural_validation_v1.json.",
    )
    parser.add_argument(
        "--cif-dir",
        type=Path,
        default=None,
        help="Directory containing CHGNet-relaxed CIFs. "
        "Defaults to convergence_source_v1/jobs/*/input_chgnet_relaxed.cif.",
    )
    parser.add_argument(
        "--gnn-checkpoint",
        type=Path,
        default=DEFAULT_GNN_CHECKPOINT,
        help="Path to the audited GNN checkpoint.",
    )
    parser.add_argument(
        "--materials-csv",
        type=Path,
        default=DEFAULT_MATERIALS_CSV,
        help="Path to the local Materials Project materials.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write ml_thermal_evaluation.csv and .json.",
    )
    args = parser.parse_args()

    if not args.validation_report.is_file():
        raise FileNotFoundError(args.validation_report)
    if not args.materials_csv.is_file():
        raise FileNotFoundError(args.materials_csv)
    if not args.gnn_checkpoint.is_file():
        raise FileNotFoundError(args.gnn_checkpoint)

    validation_report = _load_struct(args.validation_report)
    selected_candidates = validation_report.get("selected_candidates", [])
    if not selected_candidates:
        raise ValueError(
            f"No selected_candidates found in {args.validation_report}; "
            "the report must come from the structural-validation stage."
        )

    cif_dir = args.cif_dir
    primary_cif_dir: Path
    fallback_cif_dir: Path | None = None
    if cif_dir is None:
        primary_cif_dir = (
            PROJECT_ROOT
            / "research"
            / "phase_2"
            / "generation"
            / "output"
            / "w_c_dft_campaign_v1"
            / "convergence_source_v1"
            / "jobs"
        )
        fallback_cif_dir = (
            PROJECT_ROOT
            / "research"
            / "phase_2"
            / "dft_validation"
            / "candidate_cifs"
            / "w_c_structural_validation_v1"
        )
    else:
        primary_cif_dir = cif_dir

    audit_inputs = {
        "validation_report_path": str(args.validation_report),
        "validation_report_sha256": _sha256(args.validation_report),
        "cif_search_dir": str(primary_cif_dir),
        "cif_fallback_dir": str(fallback_cif_dir) if fallback_cif_dir else "",
        "gnn_checkpoint_path": str(args.gnn_checkpoint),
        "gnn_checkpoint_sha256": _sha256(args.gnn_checkpoint),
        "materials_csv_path": str(args.materials_csv),
        "materials_csv_sha256": _sha256(args.materials_csv),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn = MaterialGNN().to(device)
    state = torch.load(args.gnn_checkpoint, map_location=device, weights_only=True)
    gnn.load_state_dict(
        state["model"] if isinstance(state, dict) and "model" in state else state
    )
    gnn.eval()

    mp_grouped = _load_mp_reference(args.materials_csv)

    records: list[dict[str, Any]] = []
    for candidate in selected_candidates:
        cif_path = _select_cif_path(
            candidate["candidate_id"],
            primary_cif_dir,
            fallback_cif_dir,
        )
        record = evaluate_candidate(
            candidate,
            cif_path=cif_path,
            gnn=gnn,
            device=device,
            mp_grouped=mp_grouped,
        )
        records.append(record)

    ranked = _rank_records(records)
    csv_path, json_path = _write_outputs(
        ranked,
        validation_report_path=args.validation_report,
        cif_dir=primary_cif_dir,
        materials_csv=args.materials_csv,
        gnn_checkpoint=args.gnn_checkpoint,
        output_dir=args.output_dir,
        audit_inputs=audit_inputs,
    )

    print(f"Evaluated {len(ranked)} candidates (ML-only, no DFT).")
    print(f"  chemistry pass:       {sum(1 for r in ranked if r['chemistry']['passes_thermal_chemistry'])}")
    print(f"  competitive_with_best_mp: {sum(1 for r in ranked if r['thermal_proxy_status'] == 'competitive_with_best_mp')}")
    print(f"  output csv:  {csv_path}")
    print(f"  output json: {json_path}")
    print()
    print("Top candidates:")
    for record in ranked[:5]:
        energy = record.get("gnn_formation_energy_post_chgnet_ev_per_atom")
        energy_text = f"{energy:.3f}" if isinstance(energy, (int, float)) else "n/a"
        mp_ref = record["mp_reference"] or {}
        mp_best = mp_ref.get("best_formation_energy_per_atom")
        mp_text = f"{mp_best:.3f}" if isinstance(mp_best, (int, float)) else "n/a"
        print(
            f"  [{record['ml_rank']:>2}] {record['formula']:<12} "
            f"GNN_Ef={energy_text} eV/atom  "
            f"MP_best={mp_text} eV/atom  "
            f"status={record['thermal_proxy_status']}"
        )


if __name__ == "__main__":
    main()