"""Run reproducible multi-seed GraphVAE generation campaigns.

Each seed runs the existing ``main.py`` pipeline in an isolated subprocess.
The campaign then removes exact repeated generation hypotheses while retaining
different prototype-derived structural hypotheses for CIF-level deduplication.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from pymatgen.core import Composition


CAMPAIGN_SCHEMA_VERSION = "generation_campaign_v1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MAIN_SCRIPT = Path(__file__).resolve().parent / "main.py"
DEFAULT_VAE = PROJECT_ROOT / "research" / "phase_2" / "models" / "vae_model.pt"
DEFAULT_GNN = (
    PROJECT_ROOT
    / "research"
    / "phase_2"
    / "models"
    / "gnn_formation_energy_grouped_v1.pt"
)

BASE_MANIFEST_FIELDS = [
    "candidate_id",
    "formula",
    "prototype_uid",
    "prototype_formula",
    "num_atoms",
    "num_edges",
    "gnn_formation_energy",
    "generation_method",
    "latent_alpha",
    "oxidation_states",
    "substitutions_json",
    "num_substitutions",
    "structure_status",
    "cif_path",
]
CAMPAIGN_FIELDS = [
    "source_candidate_id",
    "source_seed",
    "source_seeds_json",
    "occurrence_count",
    "exact_hypothesis_sha256",
    "composition_key",
]


def parse_seeds(value: str) -> list[int]:
    """Parse comma-separated integers and inclusive ranges such as 42-46."""
    seeds: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token[1:]:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(f"Descending seed range is invalid: {token}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    unique = list(dict.fromkeys(seeds))
    if not unique:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return unique


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _artifact_metadata(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Required checkpoint not found: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def build_campaign_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "requirement": args.requirement,
        "max_energy": args.max_energy,
        "min_energy": args.min_energy,
        "include_elements": args.include_elements or [],
        "only_elements": args.only_elements or [],
        "domain_filter": args.domain_filter,
        "allowed_elements": args.allowed_elements or [],
        "limit": args.limit,
        "seeds": args.seeds,
        "n_samples_per_seed": args.n_samples_per_seed,
        # Never truncate per-seed output before global aggregation.
        "per_seed_top_k": args.n_samples_per_seed,
        "global_top_k": args.global_top_k,
        "interpolate": not args.no_interpolate,
        "perturb_scale": args.perturb_scale,
        "vae_checkpoint": _artifact_metadata(Path(args.vae_checkpoint)),
        "gnn_checkpoint": _artifact_metadata(Path(args.gnn_checkpoint)),
    }


def build_seed_command(
    config: dict[str, Any],
    *,
    seed: int,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--requirement",
        config["requirement"],
        "--limit",
        str(config["limit"]),
        "--n-samples",
        str(config["n_samples_per_seed"]),
        "--top-k",
        str(config["per_seed_top_k"]),
        "--seed",
        str(seed),
        "--perturb-scale",
        str(config["perturb_scale"]),
        "--vae-checkpoint",
        config["vae_checkpoint"]["path"],
        "--gnn-checkpoint",
        config["gnn_checkpoint"]["path"],
        "--output",
        str(output_dir),
    ]
    if config["max_energy"] is not None:
        command.extend(["--max-energy", str(config["max_energy"])])
    if config["min_energy"] is not None:
        command.extend(["--min-energy", str(config["min_energy"])])
    if config["include_elements"]:
        command.extend(["--include-elements", ",".join(config["include_elements"])])
    if config["only_elements"]:
        command.extend(["--only-elements", ",".join(config["only_elements"])])
    if config["domain_filter"]:
        command.extend(["--domain-filter", config["domain_filter"]])
    elif config["allowed_elements"]:
        command.extend(["--allowed-elements", ",".join(config["allowed_elements"])])
    if not config["interpolate"]:
        command.append("--no-interpolate")
    return command


def _canonical_substitutions(value: str) -> tuple[str, list[dict[str, Any]]]:
    try:
        decoded = json.loads(value or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError("invalid_substitutions_json") from exc
    if not isinstance(decoded, list):
        raise ValueError("invalid_substitutions_json:not_a_list")
    if not all(isinstance(item, dict) for item in decoded):
        raise ValueError("invalid_substitutions_json:item_not_object")
    normalized = sorted(
        decoded,
        key=lambda item: (
            int(item.get("site_index", -1)),
            int(item.get("from_Z", -1)),
            int(item.get("to_Z", -1)),
        ),
    )
    return _stable_json(normalized), normalized


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _float_energy(row: dict[str, Any]) -> float:
    try:
        return float(row["gnn_formation_energy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid gnn_formation_energy for candidate {row.get('candidate_id', '?')}"
        ) from exc


def _normalize_occurrence(row: dict[str, str], seed: int) -> dict[str, Any]:
    try:
        reduced_formula = Composition(row["formula"]).reduced_formula
        num_atoms = int(row["num_atoms"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid campaign manifest row: {row}") from exc
    canonical_substitutions, substitutions = _canonical_substitutions(
        row.get("substitutions_json", "")
    )
    exact_payload = {
        "reduced_formula": reduced_formula,
        "num_atoms": num_atoms,
        "prototype_uid": row.get("prototype_uid", ""),
        "substitutions": substitutions,
    }
    exact_hash = _sha256_json(exact_payload)
    normalized: dict[str, Any] = dict(row)
    normalized.update({
        "formula": reduced_formula,
        "num_atoms": num_atoms,
        "gnn_formation_energy": _float_energy(row),
        "substitutions_json": canonical_substitutions,
        "source_seed": seed,
        "source_candidate_id": row.get("candidate_id", ""),
        "exact_hypothesis_sha256": exact_hash,
        "composition_key": f"{reduced_formula}|{num_atoms}",
    })
    return normalized


def aggregate_campaign(
    campaign_dir: Path,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate completed seed runs and write deterministic campaign artifacts."""
    campaign_dir = Path(campaign_dir)
    occurrences: list[dict[str, Any]] = []
    summaries: dict[int, dict[str, Any]] = {}
    failed_seeds: list[int] = []
    empty_seeds: list[int] = []

    for seed in config["seeds"]:
        run_dir = campaign_dir / "runs" / f"seed_{seed}"
        manifest_path = run_dir / "generation_manifest.csv"
        summary_path = run_dir / "generation_summary.json"
        if not manifest_path.exists() or not summary_path.exists():
            failed_seeds.append(seed)
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("generation_seed", -1)) != seed:
            raise ValueError(f"Seed provenance mismatch in {summary_path}")
        summaries[seed] = summary
        rows = _read_csv(manifest_path)
        if not rows:
            empty_seeds.append(seed)
        occurrences.extend(_normalize_occurrence(row, seed) for row in rows)

    if not summaries:
        failure_summary = {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "campaign_status": "failed",
            "campaign_config_sha256": _sha256_json(config),
            "config": config,
            "successful_seed_count": 0,
            "failed_seeds": failed_seeds,
            "error": "no_completed_seed_runs",
        }
        (campaign_dir / "campaign_summary.json").write_text(
            json.dumps(failure_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        raise RuntimeError(
            "No seed run completed; campaign aggregation was not performed. "
            "Check each runs/seed_*/run.log (Neo4j must be running)."
        )

    exact_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in occurrences:
        exact_groups[row["exact_hypothesis_sha256"]].append(row)

    unique_hypotheses: list[dict[str, Any]] = []
    for exact_hash, group in exact_groups.items():
        selected = min(
            group,
            key=lambda row: (
                _float_energy(row),
                int(row["source_seed"]),
                row.get("source_candidate_id", ""),
            ),
        ).copy()
        observed_seeds = sorted({int(row["source_seed"]) for row in group})
        selected.update({
            "candidate_id": f"camp_{exact_hash[:12]}",
            "source_seeds_json": json.dumps(observed_seeds),
            "occurrence_count": len(group),
            "structure_status": "not_built",
            "cif_path": "",
        })
        unique_hypotheses.append(selected)
    unique_hypotheses.sort(
        key=lambda row: (_float_energy(row), row["formula"], row["candidate_id"])
    )

    formula_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unique_hypotheses:
        formula_groups[row["formula"]].append(row)
    formula_rows: list[dict[str, Any]] = []
    composition_shortlist: list[dict[str, Any]] = []
    for formula, variants in formula_groups.items():
        best = min(variants, key=lambda row: (_float_energy(row), row["candidate_id"]))
        raw_occurrences = [row for row in occurrences if row["formula"] == formula]
        seeds = sorted({int(row["source_seed"]) for row in raw_occurrences})
        formula_rows.append({
            "formula": formula,
            "structural_hypothesis_count": len(variants),
            "total_occurrence_count": len(raw_occurrences),
            "observed_seeds_json": json.dumps(seeds),
            "best_candidate_id": best["candidate_id"],
            "best_gnn_formation_energy": _float_energy(best),
        })
        composition_shortlist.append(best.copy())
    formula_rows.sort(key=lambda row: (row["best_gnn_formation_energy"], row["formula"]))
    composition_shortlist.sort(
        key=lambda row: (_float_energy(row), row["formula"], row["candidate_id"])
    )
    composition_shortlist = composition_shortlist[: config["global_top_k"]]

    all_occurrence_fields = BASE_MANIFEST_FIELDS + [
        "source_candidate_id", "source_seed", "exact_hypothesis_sha256", "composition_key"
    ]
    campaign_manifest_fields = BASE_MANIFEST_FIELDS + CAMPAIGN_FIELDS
    _write_csv(campaign_dir / "campaign_all_occurrences.csv", occurrences, all_occurrence_fields)
    _write_csv(
        campaign_dir / "campaign_generation_manifest.csv",
        unique_hypotheses,
        campaign_manifest_fields,
    )
    _write_csv(
        campaign_dir / "campaign_shortlist.csv",
        composition_shortlist,
        campaign_manifest_fields,
    )
    _write_csv(
        campaign_dir / "campaign_formula_summary.csv",
        formula_rows,
        [
            "formula", "structural_hypothesis_count", "total_occurrence_count",
            "observed_seeds_json", "best_candidate_id", "best_gnn_formation_energy",
        ],
    )

    rejection_counts: Counter[str] = Counter()
    total_decoded = 0
    chemistry_passed = 0
    seed_results = []
    retrieved_sets: list[tuple[int, tuple[str, ...]]] = []
    for seed, summary in summaries.items():
        chemistry = summary.get("chemistry_filter_summary", {})
        total_decoded += int(chemistry.get("generated", summary.get("n_samples_requested", 0)))
        chemistry_passed += int(chemistry.get("passed", summary.get("n_generated", 0)))
        rejection_counts.update(chemistry.get("rejection_counts", {}))
        retrieved_uids = tuple(summary.get("retrieved_uids", []))
        if retrieved_uids:
            retrieved_sets.append((seed, retrieved_uids))
        seed_results.append({
            "seed": seed,
            "n_retrieved": summary.get("n_retrieved", 0),
            "n_manifest_candidates": summary.get("n_validated", 0),
            "chemistry_passed": chemistry.get("passed", 0),
            "elapsed_seconds": summary.get("elapsed_seconds"),
        })

    energies = [_float_energy(row) for row in unique_hypotheses]
    prototype_count = len({row.get("prototype_uid", "") for row in unique_hypotheses})
    chemical_systems = {
        "-".join(sorted(element.symbol for element in Composition(row["formula"]).elements))
        for row in unique_hypotheses
    }
    reference_uids = retrieved_sets[0][1] if retrieved_sets else ()
    prototype_set_consistent = all(uids == reference_uids for _, uids in retrieved_sets)
    summary = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_status": "partial" if failed_seeds else "complete",
        "campaign_config_sha256": _sha256_json(config),
        "config": config,
        "successful_seed_count": len(summaries),
        "failed_seeds": failed_seeds,
        "empty_seeds": empty_seeds,
        "seed_results": sorted(seed_results, key=lambda item: item["seed"]),
        "prototype_set_consistent_across_seeds": prototype_set_consistent,
        "total_latent_samples_decoded": total_decoded,
        "chemistry_pass_count": chemistry_passed,
        "emitted_manifest_occurrence_count": len(occurrences),
        "unique_pre_cif_hypothesis_count": len(unique_hypotheses),
        "unique_reduced_formula_count": len(formula_groups),
        "exact_duplicate_occurrences_removed": len(occurrences) - len(unique_hypotheses),
        "same_formula_structural_variants_retained": len(unique_hypotheses) - len(formula_groups),
        "prototype_diversity_count": prototype_count,
        "chemical_system_diversity_count": len(chemical_systems),
        "chemical_systems": sorted(chemical_systems),
        "chemistry_rejection_counts": dict(rejection_counts),
        "chemistry_pass_yield": chemistry_passed / total_decoded if total_decoded else 0.0,
        "manifest_yield": len(occurrences) / total_decoded if total_decoded else 0.0,
        "unique_hypothesis_yield": len(unique_hypotheses) / total_decoded if total_decoded else 0.0,
        "unique_formula_yield": len(formula_groups) / total_decoded if total_decoded else 0.0,
        "gnn_energy_statistics_ev_per_atom": {
            "min": min(energies) if energies else None,
            "median": statistics.median(energies) if energies else None,
            "max": max(energies) if energies else None,
        },
        "global_shortlist_count": len(composition_shortlist),
        "dedup_policy": {
            "pre_cif_exact_key": (
                "reduced_formula+num_atoms+prototype_uid+canonical_site_substitutions"
            ),
            "composition_grouping": "reduced_formula (reporting/shortlist only)",
            "post_cif_required": "StructureMatcher before final structural dedup",
        },
        "next_manifest_for_cif_build": str(
            campaign_dir / "campaign_generation_manifest.csv"
        ),
    }
    (campaign_dir / "campaign_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _run_is_reusable(run_dir: Path, expected: dict[str, Any]) -> bool:
    config_path = run_dir / "campaign_run_config.json"
    manifest_path = run_dir / "generation_manifest.csv"
    summary_path = run_dir / "generation_summary.json"
    if not (config_path.exists() and manifest_path.exists() and summary_path.exists()):
        return False
    recorded = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return recorded == expected and int(summary.get("generation_seed", -1)) == expected["seed"]


def _run_seed(
    config: dict[str, Any],
    *,
    campaign_hash: str,
    seed: int,
    run_dir: Path,
    resume: bool,
    dry_run: bool,
) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    expected = {"campaign_config_sha256": campaign_hash, "seed": seed}
    if resume and _run_is_reusable(run_dir, expected):
        print(f"[seed {seed}] complete; reusing existing artifacts")
        return 0
    existing_outputs = (run_dir / "generation_manifest.csv").exists() or (
        run_dir / "generation_summary.json"
    ).exists()
    if existing_outputs:
        raise ValueError(
            f"Seed {seed} has artifacts that do not match this campaign: {run_dir}. "
            "Use a new output directory."
        )

    command = build_seed_command(config, seed=seed, output_dir=run_dir)
    (run_dir / "campaign_run_config.json").write_text(
        json.dumps(expected, indent=2), encoding="utf-8"
    )
    (run_dir / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    if dry_run:
        print(f"[seed {seed}] DRY RUN: {' '.join(command)}")
        return 0

    log_path = run_dir / "run.log"
    print(f"\n[seed {seed}] starting")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            print(f"[seed {seed}] {line}", end="")
        return_code = process.wait()
    print(f"[seed {seed}] exit={return_code}; log={log_path}")
    return return_code


def run_campaign(args: argparse.Namespace) -> dict[str, Any] | None:
    campaign_dir = args.output.resolve()
    campaign_dir.mkdir(parents=True, exist_ok=True)
    config = build_campaign_config(args)
    campaign_hash = _sha256_json(config)
    config_path = campaign_dir / "campaign_config.json"
    if config_path.exists():
        recorded = json.loads(config_path.read_text(encoding="utf-8"))
        if recorded != config:
            raise ValueError(
                f"Campaign config differs from existing {config_path}; use a new output directory."
            )
    else:
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    failures: list[int] = []
    started = time.time()
    if not args.aggregate_only:
        for seed in config["seeds"]:
            code = _run_seed(
                config,
                campaign_hash=campaign_hash,
                seed=seed,
                run_dir=campaign_dir / "runs" / f"seed_{seed}",
                resume=args.resume,
                dry_run=args.dry_run,
            )
            if code != 0:
                failures.append(seed)
                if args.fail_fast:
                    break
    if args.dry_run:
        return None

    summary = aggregate_campaign(campaign_dir, config=config)
    summary["runner_failed_seeds"] = failures
    summary["campaign_wall_time_seconds"] = round(time.time() - started, 1)
    (campaign_dir / "campaign_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nCAMPAIGN RESULT")
    print(f"  decoded samples:       {summary['total_latent_samples_decoded']}")
    print(f"  manifest occurrences:  {summary['emitted_manifest_occurrence_count']}")
    print(f"  unique hypotheses:     {summary['unique_pre_cif_hypothesis_count']}")
    print(f"  unique formulas:       {summary['unique_reduced_formula_count']}")
    print(f"  candidate manifest:    {summary['next_manifest_for_cif_build']}")
    print(f"  summary:               {campaign_dir / 'campaign_summary.json'}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirement", required=True)
    parser.add_argument("--include-elements", type=lambda s: [v.strip() for v in s.split(",") if v.strip()])
    parser.add_argument("--only-elements", type=lambda s: [v.strip() for v in s.split(",") if v.strip()])
    domain = parser.add_mutually_exclusive_group()
    domain.add_argument("--domain-filter", choices=["refractory_carbide_v1"])
    domain.add_argument("--allowed-elements", type=lambda s: [v.strip() for v in s.split(",") if v.strip()])
    parser.add_argument("--max-energy", type=float, default=None)
    parser.add_argument("--min-energy", type=float, default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("42-46"))
    parser.add_argument("--n-samples-per-seed", type=int, default=100)
    parser.add_argument("--global-top-k", type=int, default=50)
    parser.add_argument("--perturb-scale", type=float, default=0.5)
    parser.add_argument("--no-interpolate", action="store_true")
    parser.add_argument("--vae-checkpoint", default=str(DEFAULT_VAE))
    parser.add_argument("--gnn-checkpoint", default=str(DEFAULT_GNN))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    run_campaign(build_arg_parser().parse_args())
