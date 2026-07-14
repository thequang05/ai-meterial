from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch
from pymatgen.core import Lattice, Structure

from structure_builder import (
    StructureBuildError,
    apply_site_substitutions,
    candidate_atomic_numbers_from_substitutions,
    verify_graph_structure_alignment,
    write_candidate_cif,
)


def _prototype(species: list[str]) -> Structure:
    coords = [[i / len(species), 0, 0] for i in range(len(species))]
    return Structure(Lattice.cubic(5.0), species, coords)


class StructureBuilderTests(unittest.TestCase):
    def test_alignment_and_unrelaxed_cif_round_trip_for_three_prototypes(self):
        prototypes = [
            ("TaTiWC3", ["Ta", "Ti", "W", "C", "C", "C"], {0: 22}),
            ("Ti3WC4", ["Ti", "Ti", "Ti", "W", "C", "C", "C", "C"], {0: 41}),
            ("Zr4WC5", ["Zr", "Zr", "Zr", "Zr", "W", "C", "C", "C", "C", "C"], {1: 72}),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            for index, (_, species, replacements) in enumerate(prototypes):
                prototype = _prototype(species)
                original_numbers = list(prototype.atomic_numbers)
                graph = SimpleNamespace(x=torch.tensor(original_numbers).view(-1, 1))
                verify_graph_structure_alignment(graph, prototype)

                candidate_numbers = original_numbers.copy()
                for site, to_z in replacements.items():
                    candidate_numbers[site] = to_z
                candidate, substitutions = apply_site_substitutions(
                    prototype, original_numbers, candidate_numbers
                )

                # The input object must remain a reusable, unchanged prototype.
                self.assertEqual(list(prototype.atomic_numbers), original_numbers)
                self.assertEqual(len(substitutions), len(replacements))
                self.assertEqual(list(candidate.lattice.abc), list(prototype.lattice.abc))

                output = Path(temp_dir) / f"candidate_{index}.cif"
                write_candidate_cif(candidate, output)
                parsed = Structure.from_file(output)
                self.assertEqual(
                    parsed.composition.reduced_formula,
                    candidate.composition.reduced_formula,
                )

    def test_alignment_mismatch_is_rejected(self):
        prototype = _prototype(["Ta", "Ti", "W", "C"])
        graph = SimpleNamespace(x=torch.tensor([22, 73, 74, 6]).view(-1, 1))
        with self.assertRaisesRegex(StructureBuildError, "graph_structure_order_mismatch"):
            verify_graph_structure_alignment(graph, prototype)

    def test_substitution_provenance_must_match_source_site(self):
        with self.assertRaisesRegex(StructureBuildError, "substitution_source_mismatch"):
            candidate_atomic_numbers_from_substitutions(
                [73, 22, 74, 6],
                [{"site_index": 0, "from_Z": 22, "to_Z": 72}],
            )

    def test_cif_writer_relabels_substituted_duplicate_species(self):
        prototype = _prototype(["Ta", "Ti", "W", "C"])
        candidate, _ = apply_site_substitutions(
            prototype,
            prototype.atomic_numbers,
            [73, 73, 74, 6],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "unique_labels.cif"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                write_candidate_cif(candidate, output)
            self.assertFalse(
                any("Site labels are not unique" in str(item.message) for item in caught)
            )
            parsed = Structure.from_file(output)
            self.assertEqual(len(parsed.labels), len(set(parsed.labels)))


if __name__ == "__main__":
    unittest.main()
