from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure

from ml_relaxation import prediction_metrics, summarize_relaxation


def _structure(a: float = 5.6) -> Structure:
    return Structure(
        Lattice.cubic(a),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )


class MLRelaxationTests(unittest.TestCase):
    def test_prediction_metrics_uses_force_vector_norm(self):
        metrics = prediction_metrics({
            "e": np.array(-2.0),
            "f": np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]),
            "s": np.array([[1.0, -2.0], [0.0, 0.5]]),
        })
        self.assertEqual(metrics["energy_ev_per_atom"], -2.0)
        self.assertEqual(metrics["max_force_ev_per_angstrom"], 5.0)
        self.assertEqual(metrics["max_abs_stress_gpa"], 2.0)

    def test_converged_relaxation_is_labeled_ml_not_dft(self):
        initial = _structure(5.6)
        final = _structure(5.5)
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            record = summarize_relaxation(
                candidate_id="candidate",
                formula="NaCl",
                input_cif=base / "input.cif",
                output_cif=base / "output.cif",
                trajectory_path=base / "trajectory.pkl",
                initial_structure=initial,
                final_structure=final,
                initial_metrics={
                    "energy_ev_per_atom": -1.0,
                    "max_force_ev_per_angstrom": 0.2,
                    "rms_force_ev_per_angstrom": 0.1,
                    "max_abs_stress_gpa": 1.0,
                },
                final_metrics={
                    "energy_ev_per_atom": -1.1,
                    "max_force_ev_per_angstrom": 0.02,
                    "rms_force_ev_per_angstrom": 0.01,
                    "max_abs_stress_gpa": 0.2,
                },
                trajectory_frame_count=10,
                fmax=0.05,
                max_steps=500,
                relax_cell=True,
                device="cpu",
                model_name="0.3.0",
                model_version="0.4.2",
            )
        self.assertEqual(record["relaxation_status"], "converged")
        self.assertEqual(record["structure_status"], "ml_relaxed_chgnet")
        self.assertNotIn("dft", record["structure_status"])

    def test_force_above_threshold_is_not_converged(self):
        structure = _structure()
        metrics = {
            "energy_ev_per_atom": -1.0,
            "max_force_ev_per_angstrom": 0.08,
            "rms_force_ev_per_angstrom": 0.04,
            "max_abs_stress_gpa": 0.5,
        }
        record = summarize_relaxation(
            candidate_id="candidate",
            formula="NaCl",
            input_cif=Path("input.cif"),
            output_cif=Path("output.cif"),
            trajectory_path=Path("trajectory.pkl"),
            initial_structure=structure,
            final_structure=structure,
            initial_metrics=metrics,
            final_metrics=metrics,
            trajectory_frame_count=502,
            fmax=0.05,
            max_steps=500,
            relax_cell=True,
            device="cpu",
            model_name="0.3.0",
            model_version="0.4.2",
        )
        self.assertEqual(record["relaxation_status"], "not_converged")
        self.assertFalse(record["force_converged"])


if __name__ == "__main__":
    unittest.main()
