"""CHGNet pre-relaxation utilities for generated candidate structures.

CHGNet is a machine-learned interatomic potential.  Its output is useful for
screening and preparing structures for DFT, but must never be reported as a
DFT-relaxed structure or as experimental evidence of high-temperature
performance.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

from structure_audit import audit_structure


RELAXATION_WORKFLOW_VERSION = "chgnet_pre_relax_v1"


def select_device(requested: str = "auto") -> str:
    """Choose a torch device without silently claiming unavailable hardware."""
    import torch

    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available")
        if requested not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported device: {requested}")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def prediction_metrics(prediction: dict[str, Any]) -> dict[str, float]:
    """Reduce a CHGNet efsm prediction to auditable scalar metrics."""
    energy = float(_array(prediction["e"]).reshape(-1)[0])
    forces = _array(prediction["f"])
    stress = _array(prediction["s"])
    force_norms = np.linalg.norm(forces, axis=1) if forces.size else np.array([math.nan])
    return {
        "energy_ev_per_atom": energy,
        "max_force_ev_per_angstrom": float(np.max(force_norms)),
        "rms_force_ev_per_angstrom": float(np.sqrt(np.mean(forces**2))),
        "max_abs_stress_gpa": float(np.max(np.abs(stress))),
    }


def write_relaxed_cif(structure: Structure, output_path: Path) -> None:
    """Write the final CHGNet structure while preserving the original object."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean = structure.copy()
    for property_name in list(clean.site_properties):
        clean.remove_site_property(property_name)
    clean.relabel_sites()
    CifWriter(clean).write_file(str(output_path))


def summarize_relaxation(
    *,
    candidate_id: str,
    formula: str,
    input_cif: Path,
    output_cif: Path,
    trajectory_path: Path,
    initial_structure: Structure,
    final_structure: Structure,
    initial_metrics: dict[str, float],
    final_metrics: dict[str, float],
    trajectory_frame_count: int,
    fmax: float,
    max_steps: int,
    relax_cell: bool,
    device: str,
    model_name: str,
    model_version: str,
) -> dict[str, Any]:
    """Classify convergence and build one serializable provenance record."""
    final_audit = audit_structure(final_structure, expected_formula=formula)
    force_converged = (
        math.isfinite(final_metrics["max_force_ev_per_angstrom"])
        and final_metrics["max_force_ev_per_angstrom"] <= fmax * 1.001
    )
    # CHGNet records the initial frame and explicitly records the final frame;
    # report frames rather than pretending this is an exact optimizer step count.
    likely_step_limit_reached = trajectory_frame_count >= max_steps + 2
    volume_change_percent = (
        100.0 * (final_structure.volume - initial_structure.volume) / initial_structure.volume
    )
    energy_delta = (
        final_metrics["energy_ev_per_atom"] - initial_metrics["energy_ev_per_atom"]
    )

    review_flags: list[str] = []
    if abs(volume_change_percent) > 25.0:
        review_flags.append(f"large_volume_change:{volume_change_percent:.2f}%")
    if energy_delta > 1e-4:
        review_flags.append(f"ml_energy_increased:{energy_delta:.6f}_eV_per_atom")
    if likely_step_limit_reached:
        review_flags.append("optimizer_step_limit_likely_reached")
    if final_audit["audit_status"] == "warn":
        review_flags.extend(final_audit["warning_reasons"])

    if not force_converged or final_audit["audit_status"] == "fail":
        relaxation_status = "not_converged"
        structure_status = "ml_relaxation_not_converged"
    elif review_flags:
        relaxation_status = "converged_review_required"
        structure_status = "ml_relaxed_chgnet_review"
    else:
        relaxation_status = "converged"
        structure_status = "ml_relaxed_chgnet"

    return {
        "workflow_version": RELAXATION_WORKFLOW_VERSION,
        "candidate_id": candidate_id,
        "formula": formula,
        "relaxation_status": relaxation_status,
        "structure_status": structure_status,
        "force_converged": force_converged,
        "review_flags": review_flags,
        "device": device,
        "model_family": "CHGNet",
        "model_name": model_name,
        "model_version": model_version,
        "optimizer": "FIRE",
        "fmax_ev_per_angstrom": fmax,
        "max_steps": max_steps,
        "relax_cell": relax_cell,
        "trajectory_frame_count": trajectory_frame_count,
        "likely_step_limit_reached": likely_step_limit_reached,
        "input_cif": str(input_cif),
        "output_cif": str(output_cif),
        "trajectory_path": str(trajectory_path),
        "num_sites": len(final_structure),
        "initial_energy_ev_per_atom": initial_metrics["energy_ev_per_atom"],
        "final_energy_ev_per_atom": final_metrics["energy_ev_per_atom"],
        "energy_change_ev_per_atom": energy_delta,
        "initial_max_force_ev_per_angstrom": initial_metrics[
            "max_force_ev_per_angstrom"
        ],
        "final_max_force_ev_per_angstrom": final_metrics[
            "max_force_ev_per_angstrom"
        ],
        "initial_rms_force_ev_per_angstrom": initial_metrics[
            "rms_force_ev_per_angstrom"
        ],
        "final_rms_force_ev_per_angstrom": final_metrics[
            "rms_force_ev_per_angstrom"
        ],
        "initial_max_abs_stress_gpa": initial_metrics["max_abs_stress_gpa"],
        "final_max_abs_stress_gpa": final_metrics["max_abs_stress_gpa"],
        "initial_volume_a3": float(initial_structure.volume),
        "final_volume_a3": float(final_structure.volume),
        "volume_change_percent": volume_change_percent,
        "final_geometry_audit": final_audit,
        "scientific_limit": (
            "Machine-learning pre-relaxation only; not DFT, not a formation-energy "
            "calculation, and not evidence of high-temperature performance."
        ),
    }
