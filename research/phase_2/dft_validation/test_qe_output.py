from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qe_output import summarize_qe_output


class QEOutputTests(unittest.TestCase):
    def test_parses_last_energy_force_pressure_and_done_marker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "pw.out"
            output.write_text(
                """
     Program PWSCF v.7.4 starts
     convergence has been achieved in 12 iterations
!    total energy              =   -100.00000000 Ry
     Total force =     0.020000     Total SCF correction = 0.0
     P=       2.50
     bfgs converged in 3 scf cycles and 2 bfgs steps
!    total energy              =   -101.25000000 Ry
     Total force =     0.000700
     P=      -0.20
     JOB DONE.
""",
                encoding="utf-8",
            )
            result = summarize_qe_output(output)
            self.assertTrue(result["job_done"])
            self.assertTrue(result["ionic_converged_marker"])
            self.assertEqual(result["total_energy_ry"], -101.25)
            self.assertEqual(result["total_force_ry_per_bohr"], 0.0007)
            self.assertEqual(result["pressure_kbar"], -0.2)
            self.assertEqual(result["scf_cycle_count"], 2)
            self.assertEqual(result["program_version"], "7.4")

    def test_detects_non_convergence_and_fatal_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "pw.out"
            output.write_text(
                "convergence NOT achieved\nError in routine electrons (1)\n",
                encoding="utf-8",
            )
            result = summarize_qe_output(output)
            self.assertFalse(result["job_done"])
            self.assertTrue(result["electronic_convergence_failed"])
            self.assertTrue(result["fatal_error_detected"])

    def test_missing_output_returns_complete_false_schema(self):
        result = summarize_qe_output(Path("definitely_missing_pw.out"))
        self.assertFalse(result["output_exists"])
        self.assertFalse(result["ionic_converged_marker"])
        self.assertEqual(result["scf_cycle_count"], 0)
        self.assertIsNone(result["program_version"])


if __name__ == "__main__":
    unittest.main()
