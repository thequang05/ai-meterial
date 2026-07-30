"""Build unrelaxed CIFs for candidates emitted by ``main.py``.

This is deliberately a separate post-generation command: it never fabricates
coordinates from graph edges and it does not make a candidate valid merely
because a CIF was requested.  Each row gets an explicit status and rejection
reason in ``structure_manifest.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from structure_builder import (
    DEFAULT_STRUCTURE_CACHE,
    StructureBuildError,
    build_candidate_cif,
)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_GRAPH_DATA = _PROJECT_ROOT / "research" / "phase_2" / "data" / "processed" / "materials_graphs.pt"


def _load_graphs(path: Path) -> dict[str, object]:
    graphs = torch.load(path, weights_only=False)
    return {
        str(graph.material_uid): graph
        for graph in graphs
        if hasattr(graph, "material_uid")
    }


def _read_substitutions(value: str) -> list[dict]:
    if not value:
        raise StructureBuildError("missing_substitutions_json:rerun_phase2_generation")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise StructureBuildError("invalid_substitutions_json") from exc
    if not isinstance(decoded, list):
        raise StructureBuildError("invalid_substitutions_json:not_a_list")
    return decoded


def run(args: argparse.Namespace) -> int:
    manifest = Path(args.manifest)
    output_dir = Path(args.output_dir)
    cif_dir = output_dir / "cifs"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "structure_manifest.csv"

    if not manifest.exists():
        raise FileNotFoundError(f"Generation manifest not found: {manifest}")
    graphs = _load_graphs(Path(args.graph_data))

    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print("No generated candidates in manifest; no CIFs written.")
        return 0

    report_rows: list[dict[str, str | int]] = []
    for row in rows:
        candidate_id = row.get("candidate_id", "")
        prototype_uid = row.get("prototype_uid", "")
        report = {
            "candidate_id": candidate_id,
            "prototype_uid": prototype_uid,
            "candidate_formula": row.get("formula", ""),
            "structure_status": "rejected",
            "rejection_reason": "",
            "num_substitutions": "",
            "cif_path": "",
        }
        graph = graphs.get(prototype_uid)
        if graph is None:
            report["rejection_reason"] = f"prototype_graph_not_found:{prototype_uid}"
            report_rows.append(report)
            continue
        try:
            result = build_candidate_cif(
                candidate_id=candidate_id,
                prototype_uid=prototype_uid,
                graph=graph,
                substitutions=_read_substitutions(row.get("substitutions_json", "")),
                expected_formula=row.get("formula", ""),
                output_path=cif_dir / f"{candidate_id}.cif",
                structure_cache=Path(args.structure_cache),
            )
            report.update({
                "structure_status": result["structure_status"],
                "num_substitutions": result["num_substitutions"],
                "cif_path": result["cif_path"],
            })
            print(f"  + {candidate_id}: {result['candidate_formula']} -> {result['cif_path']}")
        except (StructureBuildError, FileNotFoundError, ValueError) as exc:
            report["rejection_reason"] = str(exc)
            print(f"  ! {candidate_id}: {exc}")
        report_rows.append(report)

    fields = list(report_rows[0])
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report_rows)
    accepted = sum(r["structure_status"] == "unrelaxed" for r in report_rows)
    print(f"\nCIF reconstruction: {accepted}/{len(report_rows)} accepted")
    print(f"  Report: {report_path}")
    return 0 if accepted else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruct unrelaxed Phase 2 candidate CIFs.")
    parser.add_argument("--manifest", required=True, help="Phase 2 generation_manifest.csv")
    parser.add_argument("--output-dir", required=True, help="CIFs and structure manifest output directory")
    parser.add_argument("--graph-data", default=str(DEFAULT_GRAPH_DATA))
    parser.add_argument("--structure-cache", default=str(DEFAULT_STRUCTURE_CACHE))
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_arg_parser().parse_args()))
