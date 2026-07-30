"""Pre-relax candidate CIFs with CHGNet and re-run the geometry audit."""

from __future__ import annotations

import argparse
import csv
import json
import os
import traceback
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

from pymatgen.core import Composition, Structure

from ml_relaxation import (
    RELAXATION_WORKFLOW_VERSION,
    prediction_metrics,
    select_device,
    summarize_relaxation,
    write_relaxed_cif,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _resolve_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path
    return manifest_path.parent / path


def _json_safe(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_manifest(rows: list[dict[str, Any]], output_path: Path) -> None:
    preferred = [
        "candidate_id", "formula", "relaxation_status", "structure_status",
        "force_converged", "review_flags", "model_name", "model_version", "device",
        "fmax_ev_per_angstrom", "max_steps", "trajectory_frame_count",
        "initial_energy_ev_per_atom", "final_energy_ev_per_atom",
        "energy_change_ev_per_atom", "initial_max_force_ev_per_angstrom",
        "final_max_force_ev_per_angstrom", "initial_max_abs_stress_gpa",
        "final_max_abs_stress_gpa", "volume_change_percent", "input_cif",
        "output_cif", "trajectory_path", "error",
    ]
    extra = sorted({key for row in rows for key in row} - set(preferred))
    fields = preferred + extra
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: _json_safe(row.get(key, "")) for key in fields} for row in rows
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    # Avoid slow, unwritable per-user matplotlib/font caches during CHGNet import.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ai-material-matplotlib-cache")
    try:
        import chgnet
        from chgnet.model import CHGNet, StructOptimizer
    except ImportError as exc:
        raise RuntimeError(
            "CHGNet is not installed. Run: .conda/bin/python -m pip install "
            "-r research/phase_2/generation/requirements_relaxation.txt"
        ) from exc

    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    relaxed_dir = output_dir / "relaxed_cifs"
    trajectory_dir = output_dir / "trajectories"
    record_dir = output_dir / "candidate_records"
    for directory in (relaxed_dir, trajectory_dir, record_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if args.candidate_id:
        manifest_rows = [
            row for row in manifest_rows if row.get("candidate_id") in args.candidate_id
        ]
    if not manifest_rows:
        raise ValueError("No candidates selected from the structure manifest")

    device = select_device(args.device)
    print(f"[CHGNet] device={device}; candidates={len(manifest_rows)}")
    model = CHGNet.load(
        model_name=args.model_name,
        use_device=device,
        verbose=True,
    )
    relaxer = StructOptimizer(
        model=model,
        optimizer_class="FIRE",
        use_device=device,
        on_isolated_atoms="error",
    )
    model_version = str(getattr(model, "version", chgnet.__version__))

    rows: list[dict[str, Any]] = []
    for index, manifest_row in enumerate(manifest_rows, start=1):
        candidate_id = manifest_row.get("candidate_id", "")
        formula = manifest_row.get("candidate_formula") or manifest_row.get("formula") or ""
        input_cif = _resolve_path(manifest_row.get("cif_path", ""), manifest_path)
        output_cif = relaxed_dir / f"{candidate_id}_chgnet_relaxed.cif"
        trajectory_path = trajectory_dir / f"{candidate_id}_chgnet.pkl"
        record_path = record_dir / f"{candidate_id}.json"
        print(f"\n[{index}/{len(manifest_rows)}] Relaxing {candidate_id} ({formula})")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                initial_structure = Structure.from_file(input_cif)
            expected_formula = Composition(formula).reduced_formula
            if initial_structure.composition.reduced_formula != expected_formula:
                raise ValueError(
                    "input_formula_mismatch:"
                    f"expected={expected_formula}:actual={initial_structure.composition.reduced_formula}"
                )

            initial_prediction = model.predict_structure(initial_structure, task="efs")
            initial_metrics = prediction_metrics(initial_prediction)
            result = relaxer.relax(
                initial_structure,
                fmax=args.fmax,
                steps=args.steps,
                relax_cell=not args.fix_cell,
                save_path=str(trajectory_path),
                loginterval=1,
                verbose=args.verbose_optimizer,
                assign_magmoms=False,
            )
            final_structure = result["final_structure"]
            final_prediction = model.predict_structure(final_structure, task="efs")
            final_metrics = prediction_metrics(final_prediction)
            write_relaxed_cif(final_structure, output_cif)

            record = summarize_relaxation(
                candidate_id=candidate_id,
                formula=expected_formula,
                input_cif=input_cif,
                output_cif=output_cif,
                trajectory_path=trajectory_path,
                initial_structure=initial_structure,
                final_structure=final_structure,
                initial_metrics=initial_metrics,
                final_metrics=final_metrics,
                trajectory_frame_count=len(result["trajectory"]),
                fmax=args.fmax,
                max_steps=args.steps,
                relax_cell=not args.fix_cell,
                device=device,
                model_name=args.model_name,
                model_version=model_version,
            )
            print(
                f"  {record['relaxation_status']}: "
                f"E {record['initial_energy_ev_per_atom']:.4f} -> "
                f"{record['final_energy_ev_per_atom']:.4f} eV/atom; "
                f"Fmax={record['final_max_force_ev_per_angstrom']:.4f} eV/A; "
                f"dV={record['volume_change_percent']:+.2f}%"
            )
        except Exception as exc:  # noqa: BLE001
            record = {
                "workflow_version": RELAXATION_WORKFLOW_VERSION,
                "candidate_id": candidate_id,
                "formula": formula,
                "relaxation_status": "failed",
                "structure_status": "ml_relaxation_failed",
                "force_converged": False,
                "review_flags": [],
                "device": device,
                "model_family": "CHGNet",
                "model_name": args.model_name,
                "model_version": model_version,
                "fmax_ev_per_angstrom": args.fmax,
                "max_steps": args.steps,
                "input_cif": str(input_cif),
                "output_cif": "",
                "trajectory_path": str(trajectory_path),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(f"  FAILED: {record['error']}")
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        rows.append(record)

    manifest_output = output_dir / "relaxation_manifest.csv"
    _write_manifest(rows, manifest_output)
    status_counts = Counter(row["relaxation_status"] for row in rows)
    summary = {
        "workflow_version": RELAXATION_WORKFLOW_VERSION,
        "input_manifest": str(manifest_path),
        "device": device,
        "model_family": "CHGNet",
        "model_name": args.model_name,
        "model_version": model_version,
        "fmax_ev_per_angstrom": args.fmax,
        "max_steps": args.steps,
        "relax_cell": not args.fix_cell,
        "candidate_count": len(rows),
        "status_counts": dict(status_counts),
        "dft_validated_count": 0,
        "scientific_limit": (
            "CHGNet pre-relaxation only. The energies are ML potential energies, "
            "not the GNN formation-energy target and not DFT validation."
        ),
        "candidates": rows,
    }
    summary_path = output_dir / "relaxation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nManifest: {manifest_output}")
    print(f"Summary:  {summary_path}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-id", action="append")
    parser.add_argument("--model-name", default="0.3.0", choices=["0.3.0", "r2scan"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--fix-cell", action="store_true")
    parser.add_argument("--verbose-optimizer", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
