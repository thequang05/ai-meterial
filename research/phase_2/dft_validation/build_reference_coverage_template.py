"""Build an intentionally incomplete reference-coverage audit template.

The template enumerates every non-empty chemical subsystem required by the
converged candidate static energies.  It never queries a database and never
marks coverage complete.  It is only useful for estimating the required
subsystems; it is deliberately not a valid input to ``compute_qe_hull.py``.
Official screening requires the cryptographically locked snapshot, inventory,
query receipts, and energy-collector summaries produced by the dedicated MP
inventory workflow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compute_qe_hull import _chemsys, _parse_entry, _read_csv, _subsystem_keys


def build_template(*, candidate_entries_path: Path, output_path: Path) -> dict:
    candidates = [
        _parse_entry(row, "candidate") for row in _read_csv(candidate_entries_path)
    ]
    converged = [candidate for candidate in candidates if candidate["converged"]]
    if not converged:
        raise ValueError("No dft_static_converged candidate entries were found")
    settings_hashes = {candidate["settings_hash"] for candidate in converged}
    if len(settings_hashes) != 1:
        raise ValueError(
            "Candidate entries use mixed static settings; split them before "
            "building a reference inventory"
        )
    systems: dict[str, dict] = {}
    for candidate in converged:
        elements = set(candidate["elements"])
        key = _chemsys(elements)
        if key in systems:
            continue
        systems[key] = {
            "elements": sorted(elements),
            "subsystems": {
                subsystem: {
                    "query_status": "not_queried",
                    "query_receipt_sha256": "",
                    "expected_reference_ids": [],
                }
                for subsystem in sorted(_subsystem_keys(elements))
            },
            "deduplications": [],
            "coverage_complete_assertion": False,
        }
    manifest = {
        "schema_version": "dft_reference_coverage_template_v1",
        "coverage_id": "TODO_fixed_reference_snapshot_id",
        "static_settings_hash": next(iter(settings_hashes)),
        "inventory": {
            "source_name": "TODO",
            "source_snapshot_id": "TODO",
            "inventory_sha256": "",
            "query_code_version": "TODO",
            "scope": "all structures from every non-empty subsystem",
        },
        "systems": systems,
        "template_status": "diagnostic_only_not_accepted_by_compute_qe_hull",
    }
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Coverage template: {output_path}")
    print(f"Chemical systems: {len(systems)}")
    print("Diagnostic template only; compute_qe_hull.py will not accept it.")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-entries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    build_template(candidate_entries_path=args.candidate_entries, output_path=args.output)
