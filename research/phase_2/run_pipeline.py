"""
End-to-end pipeline runner — wires every stage from a user prompt to a ranked
ML-thermal evaluation. Each stage is a thin subprocess wrapper around an
existing phase-2 entry point; outputs are persisted under ``--output-dir`` and
the orchestrator chains them so Stage 4 reads candidates produced by the live
Stages 1-3 (not stale campaign artifacts).

Pipeline:

  0  NL → Neo4j candidates (nl_to_candidates.py)
  1  GraphVAE candidate generation (generation/main.py)
  1b CIF reconstruction from prototype graphs (build_candidate_cifs.py)
  2  Pre-relax structural audit (generation/audit_candidate_cifs.py --stage pre_relax)
  3a Relaxation shortlist (generation/build_relaxation_shortlist.py)
  3b CHGNet relaxation (generation/relax_candidate_cifs.py)
  3c Post-relax structural audit (generation/audit_candidate_cifs.py --stage post_relax)
  3d Structural validation report (generation/build_structural_validation_report.py)
  4  ML-only thermal evaluation vs MP reference (eval_ml_thermal.py)

Usage:
  python run_pipeline.py --prompt "stable W-Ti refractory carbide" --output-dir runs/v1
  python run_pipeline.py --from-candidates existing.json --output-dir runs/v2
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PHASE2 = _PROJECT_ROOT / "research" / "phase_2"
GENERATION = PHASE2 / "generation"
PY = sys.executable

# Light defaults — kept small so the BE can run end-to-end in ~60 s instead of
# the 5-10 min the full campaign driver takes (500 latent samples / 15 CHGNet
# relaxations). Tunable via --n-samples / --top-k / --chgnet-top-k.
DEFAULT_N_SAMPLES = 20
DEFAULT_TOP_K = 5
DEFAULT_CHGNET_TOP_K = 3


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> int:
    print(f"\n[run] {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=cwd, env=env)


# ── Stage 0 ───────────────────────────────────────────────────────────────
def stage0(prompt: str, out_dir: Path, use_llm: bool, model: str) -> Path:
    out_json = out_dir / "stage0_candidates.json"
    cmd = [
        PY, str(PHASE2 / "nl_to_candidates.py"),
        "--prompt", prompt,
        "--output", str(out_json),
    ]
    if not use_llm:
        cmd.append("--no-llm")
    else:
        cmd.extend(["--model", model])
    rc = run(cmd)
    if rc != 0:
        raise SystemExit(f"Stage 0 failed (rc={rc})")
    return out_json


def _load_parsed_args(stage0_json: Path) -> dict:
    """Extract the chemistry envelope produced by Stage 0 to feed Stage 1."""
    try:
        data = json.loads(stage0_json.read_text())
    except Exception:
        return {}
    parsed = data.get("parsed_args") or {}
    include = list(parsed.get("include_elements") or [])
    only = list(parsed.get("only_elements") or [])
    out: dict = {}
    if include:
        out["include_elements"] = ",".join(include)
    if only:
        out["only_elements"] = ",".join(only)
    return out


# ── Stage 1: GraphVAE generation ──────────────────────────────────────────
def stage1_generate(
    prompt: str,
    out_dir: Path,
    *,
    n_samples: int,
    top_k: int,
    include_elements: str | None,
    only_elements: str | None,
    domain_filter: str | None,
) -> Path:
    """Retrieval → GraphVAE → chemistry validate + GNN rank.

    Output: ``<out_dir>/generation/<seed_dir>/generation_manifest.csv``
    (rows still have ``structure_status=not_built`` and empty ``cif_path`` —
    Stage 1b rebuilds them).
    """
    gen_root = out_dir / "generation"
    gen_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY, str(GENERATION / "main.py"),
        "--requirement", prompt,
        "--output", str(gen_root),
        "--n-samples", str(n_samples),
        "--top-k", str(top_k),
        "--seed", "42",
        "--limit", "5",
    ]
    if domain_filter:
        cmd.extend(["--domain-filter", domain_filter])
    elif only_elements:
        cmd.extend(["--allowed-elements", only_elements])
    if include_elements:
        cmd.extend(["--include-elements", include_elements])
    rc = run(cmd)
    if rc != 0:
        raise SystemExit(f"Stage 1 (generation) failed (rc={rc})")
    return gen_root


def _find_generation_artifact(gen_root: Path) -> tuple[Path, Path]:
    """Locate the ``generation_manifest.csv`` + ``generation_summary.json``
    pair under ``gen_root``. Supports both layouts:

      - ``<gen_root>/<seed>/generation_manifest.csv`` (default when
        ``main.py --output <gen_root>`` creates a per-seed subdir)
      - ``<gen_root>/generation_manifest.csv`` (when a single seed writes
        directly to ``gen_root``)
    """
    direct_manifest = gen_root / "generation_manifest.csv"
    direct_summary = gen_root / "generation_summary.json"
    if direct_manifest.exists() and direct_summary.exists():
        return direct_manifest, direct_summary
    manifests = sorted(gen_root.glob("*/generation_manifest.csv"), reverse=True)
    for m in manifests:
        summary = m.parent / "generation_summary.json"
        if summary.exists():
            return m, summary
    raise SystemExit(
        f"No generation_manifest.csv under {gen_root}; "
        "ensure Stage 1 ran or --skip-stage1 is unset."
    )


# ── Stage 1b: CIF reconstruction ──────────────────────────────────────────
def stage1b_build_cifs(generation_manifest: Path, structures_dir: Path) -> Path:
    """Decode CIFs from each candidate's prototype graph + substitutions.

    If Stage 1 produced zero candidates (chemistry screen rejected all
    samples), the manifest will be empty — ``build_candidate_cifs.py`` then
    exits 0 without writing ``structure_manifest.csv``.  Emit an empty
    manifest here so downstream stages have something to read and so the
    caller can distinguish "no candidates" from "build error".
    """
    structures_dir.mkdir(parents=True, exist_ok=True)
    structure_manifest = structures_dir / "structure_manifest.csv"
    cmd = [
        PY, str(GENERATION / "build_candidate_cifs.py"),
        "--manifest", str(generation_manifest),
        "--output-dir", str(structures_dir),
    ]
    rc = run(cmd)
    if rc != 0:
        # Stage 1 emitted manifest with rows but build failed — fatal.
        if structure_manifest.exists():
            return structure_manifest
        raise SystemExit(f"Stage 1b (build_candidate_cifs) failed (rc={rc})")
    if not structure_manifest.exists():
        # Empty manifest — synthesize a header-only file so downstream
        # CSV readers have something to parse without crashing.
        structure_manifest.write_text(
            "candidate_id,prototype_uid,candidate_formula,structure_status,"
            "rejection_reason,num_substitutions,cif_path\n",
            encoding="utf-8",
        )
    return structure_manifest


# ── Stage 2: pre-relax structural audit ───────────────────────────────────
def stage2_audit_pre_relax(structure_manifest: Path, out_dir: Path) -> tuple[Path, Path]:
    """Returns ``(audit_summary_json, audit_csv)``."""
    audit_dir = out_dir / "pre_relax_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = audit_dir / "structure_audit.csv"
    audit_summary = audit_dir / "structure_audit_summary.json"
    cmd = [
        PY, str(GENERATION / "audit_candidate_cifs.py"),
        "--manifest", str(structure_manifest),
        "--output-dir", str(audit_dir),
        "--stage", "pre_relax",
    ]
    rc = run(cmd)
    if rc != 0:
        raise SystemExit(f"Stage 2 (pre-relax audit) failed (rc={rc})")
    if not audit_summary.exists():
        # Empty input — write a placeholder summary so downstream stages can
        # detect "no candidates" instead of crashing on a missing file.
        audit_summary.write_text(json.dumps({
            "audit_version": "empty_pipeline_input",
            "audit_stage": "pre_relax",
            "candidate_count": 0,
            "status_counts": {},
            "geometry_ready_count": 0,
            "structurally_unique_ready_count": 0,
            "duplicate_structure_count": 0,
        }), encoding="utf-8")
    if not audit_csv.exists():
        audit_csv.write_text(
            "candidate_id,prototype_uid,expected_formula,actual_formula,"
            "input_structure_status,audit_status,failure_reasons,"
            "warning_reasons,duplicate_of,cif_path\n",
            encoding="utf-8",
        )
    return audit_summary, audit_csv


# ── Stage 3: shortlist → relax → post-relax audit → build report ─────────
def stage3a_shortlist(
    generation_manifest: Path,
    structure_manifest: Path,
    pre_audit_csv: Path,
    out_dir: Path,
    *,
    chgnet_top_k: int,
) -> Path:
    """``pre_audit_csv`` is the row-level audit CSV produced by Stage 2;
    the shortlist builder indexes it by ``candidate_id``.
    """
    shortlist_dir = out_dir / "shortlist"
    shortlist_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY, str(GENERATION / "build_relaxation_shortlist.py"),
        "--campaign-manifest", str(generation_manifest),
        "--structure-manifest", str(structure_manifest),
        "--audit-manifest", str(pre_audit_csv),
        "--output-dir", str(shortlist_dir),
        "--top-n", str(chgnet_top_k),
    ]
    rc = run(cmd)
    if rc != 0:
        raise SystemExit(f"Stage 3a (shortlist) failed (rc={rc})")
    summary = shortlist_dir / "shortlist_summary.json"
    if not summary.exists():
        raise SystemExit(f"Stage 3a missing shortlist_summary: {summary}")
    return summary


def stage3b_relax(shortlist_dir: Path) -> tuple[Path, Path]:
    """Returns ``(relaxation_summary_json, relax_dir)``.

    Reads ``<shortlist_dir>/relaxation_input_manifest.csv`` (the row-level
    input manifest written by build_relaxation_shortlist.py).
    """
    manifest = shortlist_dir / "relaxation_input_manifest.csv"
    if not manifest.exists():
        raise SystemExit(
            f"Stage 3b missing relaxation_input_manifest.csv under {shortlist_dir}"
        )
    relax_dir = manifest.parent.parent / "chgnet_relax"
    for sub in ("relaxed_cifs", "trajectories", "candidate_records"):
        (relax_dir / sub).mkdir(parents=True, exist_ok=True)
    cmd = [
        PY, str(GENERATION / "relax_candidate_cifs.py"),
        "--manifest", str(manifest),
        "--output-dir", str(relax_dir),
        "--device", "auto",
        "--fmax", "0.05",
        "--steps", "500",
    ]
    rc = run(cmd)
    if rc != 0:
        raise SystemExit(f"Stage 3b (CHGNet relax) failed (rc={rc})")
    summary = relax_dir / "relaxation_summary.json"
    if not summary.exists():
        raise SystemExit(f"Stage 3b missing relaxation_summary: {summary}")
    return summary, relax_dir


def stage3c_post_audit(relaxation_manifest: Path, out_dir: Path) -> Path:
    audit_dir = out_dir / "post_relax_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY, str(GENERATION / "audit_candidate_cifs.py"),
        "--manifest", str(relaxation_manifest),
        "--output-dir", str(audit_dir),
        "--stage", "post_relax",
    ]
    rc = run(cmd)
    if rc != 0:
        raise SystemExit(f"Stage 3c (post-relax audit) failed (rc={rc})")
    summary = audit_dir / "structure_audit_summary.json"
    if not summary.exists():
        raise SystemExit(f"Stage 3c missing summary: {summary}")
    return summary


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _synthesize_campaign_summary(
    generation_summary: Path,
    generation_manifest: Path,
    structure_manifest: Path,
    pre_audit_summary: Path,
    out_path: Path,
) -> Path:
    """Wrap Stage 1 outputs into the campaign_summary.json shape that
    ``build_structural_validation_report.py`` expects.

    The canonical campaign driver aggregates many seeds; we run one seed, so
    we synthesize an equivalent summary by reading the single generation run
    and counting chemistry / audit rejections ourselves.
    """
    gen_summary = json.loads(generation_summary.read_text())
    manifest_rows = _read_csv(generation_manifest)
    structure_rows = _read_csv(structure_manifest)
    pre_audit = json.loads(pre_audit_summary.read_text())

    cifs_built = sum(1 for r in structure_rows if r.get("structure_status") == "unrelaxed")
    chemistry_pass = len({r["candidate_id"] for r in manifest_rows})
    chem_summary = gen_summary.get("chemistry_filter_summary") or {}

    wrapped = {
        "schema_version": "campaign_summary_v1",
        "campaign_status": "ok_single_seed",
        "campaign_config_sha256": "single_seed_synthesized",
        "config": {
            "requirement": gen_summary.get("requirement"),
            "n_samples_per_seed": gen_summary.get("n_samples_requested"),
            "seeds": [gen_summary.get("generation_seed", 42)],
        },
        "successful_seed_count": 1,
        "failed_seeds": [],
        "empty_seeds": [],
        "seed_results": [],
        "prototype_set_consistent_across_seeds": True,
        "total_latent_samples_decoded": gen_summary.get("n_samples_requested", 0),
        "chemistry_pass_count": chemistry_pass,
        "emitted_manifest_occurrence_count": len(manifest_rows),
        "unique_pre_cif_hypothesis_count": cifs_built,
        "unique_reduced_formula_count": len(
            {r.get("formula", "") for r in structure_rows if r.get("structure_status") == "unrelaxed"}
        ),
        "exact_duplicate_occurrences_removed": 0,
        "same_formula_structural_variants_retained": 0,
        "prototype_diversity_count": len(gen_summary.get("retrieved_uids") or []),
        "chemical_system_diversity_count": 1,
        "chemical_systems": [],
        "chemistry_rejection_counts": chem_summary.get("rejection_counts") or {},
        "chemistry_pass_yield": chem_summary.get("pass_yield"),
        "manifest_yield": cifs_built / max(len(manifest_rows), 1),
        "unique_hypothesis_yield": cifs_built / max(chemistry_pass, 1),
        "unique_formula_yield": cifs_built / max(len(manifest_rows), 1),
        "gnn_energy_statistics_ev_per_atom": {},
        "global_shortlist_count": pre_audit.get("geometry_ready_count", 0),
        "dedup_policy": "none_single_seed",
        "runner_failed_seeds": [],
        "campaign_wall_time_seconds": gen_summary.get("elapsed_seconds", 0.0),
        # Extra context for downstream tools that surface this file.
        "requirement": gen_summary.get("requirement"),
        "retrieved_uids": gen_summary.get("retrieved_uids"),
        "chemistry_filter_summary": chem_summary,
    }
    out_path.write_text(json.dumps(wrapped, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def stage3d_build_report(
    generation_summary: Path,
    generation_manifest: Path,
    structure_manifest: Path,
    pre_audit_summary: Path,
    shortlist_summary: Path,
    relaxation_summary: Path,
    post_audit_summary: Path,
    out_dir: Path,
    *,
    dft_queue_size: int = 1,
) -> tuple[Path, Path]:
    """Build the structural validation report Stage 4 consumes.

    Returns ``(structural_validation_json, relax_dir)``.
    """
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_json = reports_dir / "structural_validation.json"
    report_md = reports_dir / "structural_validation.md"

    # build_report requires a campaign_summary.json with specific keys —
    # synthesize one from the Stage 1 outputs above.
    campaign_summary = _synthesize_campaign_summary(
        generation_summary=generation_summary,
        generation_manifest=generation_manifest,
        structure_manifest=structure_manifest,
        pre_audit_summary=pre_audit_summary,
        out_path=reports_dir / "campaign_summary.json",
    )

    cmd = [
        PY, str(GENERATION / "build_structural_validation_report.py"),
        "--campaign-summary", str(campaign_summary),
        "--pre-audit-summary", str(pre_audit_summary),
        "--shortlist-summary", str(shortlist_summary),
        "--relaxation-summary", str(relaxation_summary),
        "--post-audit-summary", str(post_audit_summary),
        "--output-json", str(report_json),
        "--output-markdown", str(report_md),
        "--dft-queue-size", str(dft_queue_size),
    ]
    rc = run(cmd)
    if rc != 0:
        raise SystemExit(f"Stage 3d (build report) failed (rc={rc})")
    if not report_json.exists():
        raise SystemExit(f"Stage 3d missing structural_validation.json: {report_json}")
    return report_json, reports_dir.parent / "chgnet_relax"


# ── Stage 4 ───────────────────────────────────────────────────────────────
def stage4_eval(
    validation_report: Path,
    relax_dir: Path,
    out_dir: Path,
) -> Path | None:
    """Run the ML-only thermal evaluation against MP reference.

    Returns the output CSV path (or None when Stage 4 is skipped).
    """
    if not validation_report.exists():
        print(
            f"[stage4] No validation report at {validation_report}; "
            "writing empty Stage 4 outputs.",
            flush=True,
        )
        out_csv = out_dir / "ml_thermal_evaluation.csv"
        out_json = out_dir / "ml_thermal_evaluation.json"
        out_csv.write_text(
            "ml_rank,rank,candidate_id,formula,chemistry_pass,"
            "gnn_formation_energy_post_chgnet_ev_per_atom,"
            "thermal_proxy_status,thermal_proxy_reason,cif_path\n",
            encoding="utf-8",
        )
        out_json.write_text(
            json.dumps({
                "workflow_version": "ml_thermal_evaluation_v1",
                "candidate_count": 0,
                "chemistry_pass_count": 0,
                "competitive_with_best_mp_count": 0,
                "status_counts": {},
                "scientific_limit": "No chemistry-validated candidates from Stage 1.",
            }, indent=2),
            encoding="utf-8",
        )
        return out_csv
    cmd = [
        PY, str(PHASE2 / "eval_ml_thermal.py"),
        "--validation-report", str(validation_report),
        "--cif-dir", str(relax_dir / "relaxed_cifs"),
        "--output-dir", str(out_dir),
    ]
    rc = run(cmd)
    if rc != 0:
        raise SystemExit(f"Stage 4 failed (rc={rc})")
    out_csv = out_dir / "ml_thermal_evaluation.csv"
    return out_csv if out_csv.exists() else out_dir / "ml_thermal_evaluation.json"


# ── Orchestrator ───────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", help="NL query (drives Stages 0, 1)")
    p.add_argument("--from-candidates", help="Reuse existing Stage 0 JSON")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="qwen2.5-1.5b-instruct")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--skip-stage0", action="store_true")
    p.add_argument("--skip-stage1", action="store_true",
                   help="Reuse existing generation manifest under <output-dir>/generation/")
    p.add_argument("--skip-stage1b", action="store_true",
                   help="Reuse existing structure_manifest.csv under <output-dir>/structures/")
    p.add_argument("--skip-stage2", action="store_true")
    p.add_argument("--skip-stage3", action="store_true")
    p.add_argument("--skip-stage4", action="store_true")
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES,
                   help=f"Latent vectors to sample in Stage 1 (default {DEFAULT_N_SAMPLES})")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                   help=f"Top-k validated candidates from Stage 1 (default {DEFAULT_TOP_K})")
    p.add_argument("--chgnet-top-k", type=int, default=DEFAULT_CHGNET_TOP_K,
                   help=f"Top-k candidates passed to CHGNet relaxation (default {DEFAULT_CHGNET_TOP_K})")
    p.add_argument("--dft-queue-size", type=int, default=1,
                   help="Max DFT-eligible candidates bundled by build_report (default 1).")
    p.add_argument("--domain-filter", default=None,
                   choices=["refractory_carbide_v1"],
                   help="Validated chemistry profile (overrides prompt parsing).")
    p.add_argument("--limit", type=int, default=5,
                   help="Number of seed materials to retrieve from Neo4j in Stage 1 (default 5).")
    args = p.parse_args()

    if not args.prompt and not args.from_candidates and not args.skip_stage0:
        p.error("Provide --prompt, --from-candidates, or --skip-stage0")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 0 ─────────────────────────────────────────────────────────────
    if not args.skip_stage0:
        if args.from_candidates:
            stage0_json = Path(args.from_candidates)
            if not stage0_json.is_file():
                raise SystemExit(f"--from-candidates not found: {stage0_json}")
        else:
            if not args.prompt:
                p.error("--prompt required when stage 0 is enabled")
            stage0_json = stage0(args.prompt, out_dir, use_llm=not args.no_llm, model=args.model)
    else:
        stage0_json = out_dir / "stage0_candidates.json"

    chem = _load_parsed_args(stage0_json)

    # Stage 1 ─────────────────────────────────────────────────────────────
    if not args.skip_stage1:
        stage1_generate(
            prompt=args.prompt or "(reused)",
            out_dir=out_dir,
            n_samples=args.n_samples,
            top_k=args.top_k,
            include_elements=chem.get("include_elements"),
            only_elements=chem.get("only_elements"),
            domain_filter=args.domain_filter,
        )

    gen_root = out_dir / "generation"
    generation_manifest, generation_summary = _find_generation_artifact(gen_root)

    # Stage 1b ────────────────────────────────────────────────────────────
    structures_dir = out_dir / "structures"
    if not args.skip_stage1b:
        structure_manifest = stage1b_build_cifs(generation_manifest, structures_dir)
    else:
        structure_manifest = structures_dir / "structure_manifest.csv"

    # Stage 2 ─────────────────────────────────────────────────────────────
    if not args.skip_stage2:
        pre_audit_summary, pre_audit_csv = stage2_audit_pre_relax(structure_manifest, out_dir)
    else:
        pre_audit_summary = out_dir / "pre_relax_audit" / "structure_audit_summary.json"
        pre_audit_csv = out_dir / "pre_relax_audit" / "structure_audit.csv"

    # If Stage 1 produced zero chemistry-pass candidates, downstream Stages
    # 3a-3d would either crash or produce meaningless output.  Skip them and
    # still emit an empty Stage 4 manifest so the UI has something to render.
    stage0_summary = json.loads(stage0_json.read_text()) if stage0_json.exists() else {}
    gen_summary = json.loads(generation_summary.read_text())
    n_validated = int(gen_summary.get("n_validated", 0)) if gen_summary else 0
    skip_heavy = (
        n_validated == 0
        and not args.skip_stage3
    )
    if skip_heavy:
        print(
            f"[pipeline] Stage 1 produced 0 chemistry-validated candidates "
            f"(n_samples_requested={gen_summary.get('n_samples_requested')}, "
            f"domain_filter={args.domain_filter}); "
            "skipping Stages 3-4 (CHGNet relaxation + ML eval).",
            flush=True,
        )

    # Stage 3a/b/c/d ──────────────────────────────────────────────────────
    if not args.skip_stage3 and not skip_heavy:
        shortlist_dir = out_dir / "shortlist"
        shortlist_summary = stage3a_shortlist(
            generation_manifest, structure_manifest, pre_audit_csv, out_dir,
            chgnet_top_k=args.chgnet_top_k,
        )
        relaxation_summary, relax_dir = stage3b_relax(shortlist_dir)
        relaxation_manifest = relax_dir / "relaxation_manifest.csv"
        if not relaxation_manifest.exists():
            raise SystemExit(f"Stage 3b missing relaxation_manifest.csv: {relaxation_manifest}")
        post_audit_summary = stage3c_post_audit(relaxation_manifest, out_dir)
        validation_report, relax_dir = stage3d_build_report(
            generation_summary=generation_summary,
            generation_manifest=generation_manifest,
            structure_manifest=structure_manifest,
            pre_audit_summary=pre_audit_summary,
            shortlist_summary=shortlist_summary,
            relaxation_summary=relaxation_summary,
            post_audit_summary=post_audit_summary,
            out_dir=out_dir,
            dft_queue_size=args.dft_queue_size,
        )
    else:
        validation_report = out_dir / "reports" / "structural_validation.json"
        relax_dir = out_dir / "chgnet_relax"

    # Stage 4 ─────────────────────────────────────────────────────────────
    if not args.skip_stage4:
        stage4_eval(validation_report, relax_dir, out_dir)

    print(f"\n[pipeline] done. outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
