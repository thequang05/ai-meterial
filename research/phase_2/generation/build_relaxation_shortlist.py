"""Build a provenance-preserving CHGNet shortlist after CIF geometry audit.

This stage deliberately keeps three decisions separate:

1. Reject geometry failures and StructureMatcher-equivalent duplicates.
2. Keep the lowest-GNN-energy structural hypothesis per reduced formula.
3. Apply the requested relaxation budget to those formula representatives.

The GNN score is only a screening score.  The output is an input manifest for
ML pre-relaxation, not a claim of DFT or experimental validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from pymatgen.core import Composition


SHORTLIST_VERSION = "campaign_relaxation_shortlist_v1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def _index_unique(rows: list[dict[str, str]], *, source: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        candidate_id = row.get("candidate_id", "").strip()
        if not candidate_id:
            raise ValueError(f"{source} contains a row without candidate_id")
        if candidate_id in indexed:
            raise ValueError(f"{source} contains duplicate candidate_id={candidate_id}")
        indexed[candidate_id] = row
    return indexed


def _energy(row: dict[str, str]) -> float:
    raw = row.get("gnn_formation_energy", "")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid gnn_formation_energy for {row.get('candidate_id', '?')}: {raw!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Non-finite gnn_formation_energy for {row.get('candidate_id', '?')}: {raw!r}"
        )
    return value


def _formula_key(row: dict[str, str]) -> str:
    formula = (row.get("formula") or row.get("candidate_formula") or "").strip()
    if not formula:
        raise ValueError(f"Missing formula for {row.get('candidate_id', '?')}")
    return Composition(formula).reduced_formula


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred: list[str]) -> None:
    extras = sorted({key for row in rows for key in row} - set(preferred))
    fields = preferred + extras
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def build_shortlist(
    *,
    campaign_manifest: Path,
    structure_manifest: Path,
    audit_manifest: Path,
    output_dir: Path,
    top_n: int = 15,
) -> dict[str, Any]:
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    campaign_manifest = Path(campaign_manifest).resolve()
    structure_manifest = Path(structure_manifest).resolve()
    audit_manifest = Path(audit_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    campaign_rows = _read_csv(campaign_manifest)
    structure_index = _index_unique(
        _read_csv(structure_manifest), source="structure manifest"
    )
    audit_index = _index_unique(_read_csv(audit_manifest), source="audit manifest")

    decisions: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for campaign_order, campaign in enumerate(campaign_rows, start=1):
        candidate_id = campaign.get("candidate_id", "").strip()
        structure = structure_index.get(candidate_id)
        audit = audit_index.get(candidate_id)
        formula_key = _formula_key(campaign)
        energy = _energy(campaign)

        decision: dict[str, Any] = dict(campaign)
        decision.update({
            "campaign_order": campaign_order,
            "formula_key": formula_key,
            "candidate_formula": formula_key,
            "gnn_formation_energy": energy,
            "cif_path": structure.get("cif_path", "") if structure else "",
            "structure_status": structure.get("structure_status", "") if structure else "",
            "audit_status": audit.get("audit_status", "") if audit else "",
            "audit_warning_reasons": audit.get("warning_reasons", "") if audit else "",
            "audit_failure_reasons": audit.get("failure_reasons", "") if audit else "",
            "duplicate_of": audit.get("duplicate_of", "") if audit else "",
            "selection_rank": "",
            "selected_formula_representative": "",
        })

        if structure is None:
            decision.update(selection_status="excluded", selection_reason="missing_structure_row")
        elif audit is None:
            decision.update(selection_status="excluded", selection_reason="missing_audit_row")
        elif structure.get("structure_status") != "unrelaxed":
            decision.update(
                selection_status="excluded",
                selection_reason=f"unexpected_structure_status:{structure.get('structure_status', '')}",
            )
        elif audit.get("audit_status") == "fail":
            decision.update(selection_status="excluded", selection_reason="geometry_audit_failed")
        elif audit.get("duplicate_of", "").strip():
            decision.update(
                selection_status="excluded",
                selection_reason="structurematcher_equivalent",
                selected_formula_representative=audit.get("duplicate_of", ""),
            )
        else:
            decision.update(selection_status="eligible", selection_reason="eligible_unique_structure")
            eligible.append(decision)
        decisions.append(decision)

    eligible.sort(
        key=lambda row: (
            float(row["gnn_formation_energy"]),
            int(row["campaign_order"]),
            str(row["candidate_id"]),
        )
    )

    representatives: list[dict[str, Any]] = []
    representative_by_formula: dict[str, dict[str, Any]] = {}
    for row in eligible:
        formula_key = str(row["formula_key"])
        representative = representative_by_formula.get(formula_key)
        if representative is not None:
            row.update(
                selection_status="excluded",
                selection_reason="higher_energy_same_formula",
                selected_formula_representative=representative["candidate_id"],
            )
            continue
        representative_by_formula[formula_key] = row
        representatives.append(row)

    selected = representatives[:top_n]
    for rank, row in enumerate(selected, start=1):
        row.update(
            selection_status="selected",
            selection_reason="selected_for_chgnet",
            selection_rank=rank,
            selected_formula_representative=row["candidate_id"],
        )
    for row in representatives[top_n:]:
        row.update(
            selection_status="excluded",
            selection_reason="outside_relaxation_budget",
            selected_formula_representative=row["candidate_id"],
        )

    decisions.sort(
        key=lambda row: (
            float(row["gnn_formation_energy"]),
            int(row["campaign_order"]),
        )
    )
    decision_path = output_dir / "shortlist_decisions.csv"
    _write_csv(
        decision_path,
        decisions,
        [
            "candidate_id", "formula", "formula_key", "candidate_formula",
            "gnn_formation_energy", "selection_status", "selection_reason",
            "selection_rank", "selected_formula_representative", "prototype_uid",
            "source_seed", "source_seeds_json", "occurrence_count", "substitutions_json",
            "audit_status", "audit_warning_reasons", "audit_failure_reasons",
            "duplicate_of", "structure_status", "cif_path", "campaign_order",
        ],
    )

    relaxation_rows: list[dict[str, Any]] = []
    for row in selected:
        relaxation_rows.append({
            "candidate_id": row["candidate_id"],
            "candidate_formula": row["formula_key"],
            "formula": row["formula_key"],
            "structure_status": row["structure_status"],
            "cif_path": row["cif_path"],
            "selection_rank": row["selection_rank"],
            "gnn_formation_energy": row["gnn_formation_energy"],
            "prototype_uid": row.get("prototype_uid", ""),
            "prototype_formula": row.get("prototype_formula", ""),
            "generation_method": row.get("generation_method", ""),
            "latent_alpha": row.get("latent_alpha", ""),
            "substitutions_json": row.get("substitutions_json", ""),
            "source_candidate_id": row.get("source_candidate_id", ""),
            "source_seed": row.get("source_seed", ""),
            "source_seeds_json": row.get("source_seeds_json", ""),
            "occurrence_count": row.get("occurrence_count", ""),
            "audit_status": row["audit_status"],
            "audit_warning_reasons": row["audit_warning_reasons"],
            "shortlist_version": SHORTLIST_VERSION,
        })
    relaxation_path = output_dir / "relaxation_input_manifest.csv"
    _write_csv(
        relaxation_path,
        relaxation_rows,
        [
            "candidate_id", "candidate_formula", "formula", "structure_status",
            "cif_path", "selection_rank", "gnn_formation_energy", "prototype_uid",
            "prototype_formula", "generation_method", "latent_alpha",
            "substitutions_json", "source_candidate_id", "source_seed",
            "source_seeds_json", "occurrence_count", "audit_status",
            "audit_warning_reasons", "shortlist_version",
        ],
    )

    reason_counts = Counter(str(row["selection_reason"]) for row in decisions)
    summary = {
        "shortlist_version": SHORTLIST_VERSION,
        "campaign_manifest": str(campaign_manifest),
        "structure_manifest": str(structure_manifest),
        "audit_manifest": str(audit_manifest),
        "candidate_count": len(campaign_rows),
        "geometry_audit_fail_count": reason_counts.get("geometry_audit_failed", 0),
        "structurematcher_equivalent_count": reason_counts.get(
            "structurematcher_equivalent", 0
        ),
        "eligible_unique_structure_count": len(eligible),
        "formula_representative_count": len(representatives),
        "relaxation_budget": top_n,
        "selected_count": len(selected),
        "selection_reason_counts": dict(sorted(reason_counts.items())),
        "selected_gnn_energy_range_ev_per_atom": {
            "min": min((float(row["gnn_formation_energy"]) for row in selected), default=None),
            "max": max((float(row["gnn_formation_energy"]) for row in selected), default=None),
        },
        "selected_candidates": [
            {
                "rank": row["selection_rank"],
                "candidate_id": row["candidate_id"],
                "formula": row["formula_key"],
                "gnn_formation_energy_ev_per_atom": row["gnn_formation_energy"],
                "prototype_uid": row.get("prototype_uid", ""),
                "source_seeds": row.get("source_seeds_json", ""),
            }
            for row in selected
        ],
        "outputs": {
            "decisions_csv": str(decision_path),
            "relaxation_input_manifest_csv": str(relaxation_path),
        },
        "scientific_limit": (
            "The GNN formation-energy values are ranking predictions only. "
            "Selection and subsequent CHGNet relaxation are not DFT or thermal validation."
        ),
    }
    summary_path = output_dir / "shortlist_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Shortlist version: {SHORTLIST_VERSION}")
    print(
        f"Candidates={len(campaign_rows)}; geometry_fail="
        f"{summary['geometry_audit_fail_count']}; equivalent="
        f"{summary['structurematcher_equivalent_count']}; unique_structures="
        f"{summary['eligible_unique_structure_count']}; formulas="
        f"{summary['formula_representative_count']}"
    )
    print(f"Selected for CHGNet: {len(selected)}/{top_n}")
    for row in selected:
        print(
            f"{int(row['selection_rank']):2d}. {row['candidate_id']} "
            f"{row['formula_key']:<16} GNN={float(row['gnn_formation_energy']):+.4f} eV/atom"
        )
    print(f"Relaxation manifest: {relaxation_path}")
    print(f"Decision audit:      {decision_path}")
    print(f"Summary:             {summary_path}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=15)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    build_shortlist(
        campaign_manifest=args.campaign_manifest,
        structure_manifest=args.structure_manifest,
        audit_manifest=args.audit_manifest,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )
