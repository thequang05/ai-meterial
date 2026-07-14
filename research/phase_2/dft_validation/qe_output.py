"""Small, dependency-light parser for Quantum ESPRESSO stdout gates."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


def _float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def summarize_qe_output(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return {
            "output_exists": False,
            "job_done": False,
            "electronic_convergence_failed": False,
            "ionic_converged_marker": False,
            "fatal_error_detected": False,
            "fatal_error_patterns": [],
            "total_energy_ry": None,
            "total_force_ry_per_bohr": None,
            "pressure_kbar": None,
            "scf_cycle_count": 0,
            "last_scf_iteration_count": None,
            "program_version": None,
        }
    text = path.read_text(encoding="utf-8", errors="replace")
    energies = re.findall(rf"!\s+total energy\s*=\s*({_NUMBER})\s+Ry", text)
    forces = re.findall(rf"Total force\s*=\s*({_NUMBER})", text)
    pressures = re.findall(rf"P=\s*({_NUMBER})", text)
    scf_iterations = [
        int(value)
        for value in re.findall(
            r"convergence has been achieved in\s+(\d+)\s+iterations", text, re.I
        )
    ]
    fatal_patterns = [
        r"Error in routine",
        r"Maximum CPU time exceeded",
        r"stopping.*error",
        r"MPI_ABORT",
        r"mpirun.*aborted",
        r"killed:\s*9",
        r"out of memory",
        r"oom-kill",
        r"segmentation fault",
        r"received signal",
    ]
    matched_fatal_patterns = [
        pattern for pattern in fatal_patterns if re.search(pattern, text, re.I)
    ]
    version_match = re.search(
        r"Program\s+PWSCF\s+v\.?\s*([^\s]+)", text, re.I
    )
    return {
        "output_exists": True,
        "job_done": "JOB DONE." in text,
        "electronic_convergence_failed": bool(
            re.search(r"convergence\s+NOT\s+achieved", text, re.I)
        ),
        "ionic_converged_marker": bool(
            re.search(r"bfgs converged|End of BFGS Geometry Optimization", text, re.I)
        ),
        "fatal_error_detected": bool(matched_fatal_patterns),
        "fatal_error_patterns": matched_fatal_patterns,
        "total_energy_ry": _float(energies[-1]) if energies else None,
        "total_force_ry_per_bohr": _float(forces[-1]) if forces else None,
        "pressure_kbar": _float(pressures[-1]) if pressures else None,
        "scf_cycle_count": len(energies),
        "last_scf_iteration_count": scf_iterations[-1] if scf_iterations else None,
        "program_version": version_match.group(1) if version_match else None,
    }
