from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from pymatgen.core import Lattice, Structure

from audit_candidate_cifs import audit_manifest
from structure_audit import audit_cif, audit_structure, find_duplicate_structures


class StructureAuditTests(unittest.TestCase):
    def test_valid_structure_passes_geometry_gate(self):
        structure = Structure(
            Lattice.cubic(5.6),
            ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        result = audit_structure(structure, expected_formula="NaCl")
        self.assertEqual(result["audit_status"], "pass")
        self.assertEqual(result["actual_formula"], "NaCl")
        self.assertGreater(result["min_distance_angstrom"], 1.0)

    def test_overlapping_sites_fail(self):
        structure = Structure(
            Lattice.cubic(5.0),
            ["C", "C"],
            [[0, 0, 0], [0, 0, 0]],
        )
        result = audit_structure(structure, expected_formula="C")
        self.assertEqual(result["audit_status"], "fail")
        self.assertTrue(any("atomic_overlap" in reason for reason in result["failure_reasons"]))

    def test_formula_mismatch_fails(self):
        structure = Structure(
            Lattice.cubic(5.6),
            ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        result = audit_structure(structure, expected_formula="KCl")
        self.assertEqual(result["audit_status"], "fail")
        self.assertTrue(any("formula_mismatch" in reason for reason in result["failure_reasons"]))

    def test_cif_round_trip_and_missing_file(self):
        structure = Structure(
            Lattice.cubic(5.6),
            ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            cif_path = Path(temp_dir) / "valid.cif"
            structure.to(filename=cif_path)
            self.assertEqual(
                audit_cif(cif_path, expected_formula="NaCl")["audit_status"],
                "pass",
            )
            self.assertEqual(
                audit_cif(Path(temp_dir) / "missing.cif")["audit_status"],
                "fail",
            )

    def test_duplicate_detection_keeps_first_candidate(self):
        first = Structure(
            Lattice.cubic(5.6),
            ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        second = first.copy()
        duplicates = find_duplicate_structures([("first", first), ("second", second)])
        self.assertEqual(duplicates, {"second": "first"})

    def test_manifest_ready_count_excludes_structural_duplicates(self):
        structure = Structure(
            Lattice.cubic(5.6),
            ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_cif = root / "first.cif"
            second_cif = root / "second.cif"
            structure.to(filename=first_cif)
            structure.to(filename=second_cif)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "candidate_id", "candidate_formula", "structure_status", "cif_path",
                    ],
                )
                writer.writeheader()
                writer.writerows([
                    {
                        "candidate_id": "first",
                        "candidate_formula": "NaCl",
                        "structure_status": "unrelaxed",
                        "cif_path": str(first_cif),
                    },
                    {
                        "candidate_id": "second",
                        "candidate_formula": "NaCl",
                        "structure_status": "unrelaxed",
                        "cif_path": str(second_cif),
                    },
                ])

            summary = audit_manifest(manifest, root / "audit")

            self.assertEqual(summary["geometry_ready_count"], 2)
            self.assertEqual(summary["duplicate_structure_count"], 1)
            self.assertEqual(summary["structurally_unique_ready_count"], 1)
            self.assertEqual(summary["ready_for_relaxation_count"], 1)
            self.assertEqual(summary["audit_stage"], "pre_relax")


if __name__ == "__main__":
    unittest.main()
