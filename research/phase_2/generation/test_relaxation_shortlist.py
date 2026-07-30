from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from build_relaxation_shortlist import build_shortlist


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class RelaxationShortlistTests(unittest.TestCase):
    def test_filters_audit_duplicate_groups_formula_and_applies_budget(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            campaign = root / "campaign.csv"
            structures = root / "structures.csv"
            audit = root / "audit.csv"
            ids = ["best_tic", "other_tic", "duplicate_wc", "failed_tac", "nbc", "moc"]
            formulas = ["TiC", "TiC", "WC", "TaC", "NbC", "MoC"]
            energies = [-1.0, -0.8, -0.7, -0.6, -0.5, -0.4]
            _write(
                campaign,
                [
                    {
                        "candidate_id": candidate_id,
                        "formula": formula,
                        "gnn_formation_energy": str(energy),
                        "prototype_uid": f"proto_{candidate_id}",
                        "source_seeds_json": "[42]",
                        "substitutions_json": "[]",
                    }
                    for candidate_id, formula, energy in zip(ids, formulas, energies)
                ],
            )
            _write(
                structures,
                [
                    {
                        "candidate_id": candidate_id,
                        "candidate_formula": formula,
                        "structure_status": "unrelaxed",
                        "cif_path": str(root / f"{candidate_id}.cif"),
                    }
                    for candidate_id, formula in zip(ids, formulas)
                ],
            )
            _write(
                audit,
                [
                    {
                        "candidate_id": candidate_id,
                        "audit_status": "fail" if candidate_id == "failed_tac" else (
                            "warn" if candidate_id == "duplicate_wc" else "pass"
                        ),
                        "duplicate_of": "original_wc" if candidate_id == "duplicate_wc" else "",
                        "warning_reasons": "[]",
                        "failure_reasons": "[]",
                    }
                    for candidate_id in ids
                ],
            )

            summary = build_shortlist(
                campaign_manifest=campaign,
                structure_manifest=structures,
                audit_manifest=audit,
                output_dir=root / "out",
                top_n=2,
            )

            self.assertEqual(summary["structurematcher_equivalent_count"], 1)
            self.assertEqual(summary["geometry_audit_fail_count"], 1)
            self.assertEqual(summary["eligible_unique_structure_count"], 4)
            self.assertEqual(summary["formula_representative_count"], 3)
            self.assertEqual(summary["selected_count"], 2)
            self.assertEqual(
                [row["candidate_id"] for row in summary["selected_candidates"]],
                ["best_tic", "nbc"],
            )

            with (root / "out" / "relaxation_input_manifest.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                relaxation_rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["candidate_id"] for row in relaxation_rows],
                ["best_tic", "nbc"],
            )

            with (root / "out" / "shortlist_decisions.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                decisions = {row["candidate_id"]: row for row in csv.DictReader(handle)}
            self.assertEqual(decisions["other_tic"]["selection_reason"], "higher_energy_same_formula")
            self.assertEqual(decisions["duplicate_wc"]["selection_reason"], "structurematcher_equivalent")
            self.assertEqual(decisions["failed_tac"]["selection_reason"], "geometry_audit_failed")
            self.assertEqual(decisions["moc"]["selection_reason"], "outside_relaxation_budget")

    def test_rejects_non_positive_budget(self):
        with self.assertRaisesRegex(ValueError, "top_n"):
            build_shortlist(
                campaign_manifest=Path("missing"),
                structure_manifest=Path("missing"),
                audit_manifest=Path("missing"),
                output_dir=Path("missing"),
                top_n=0,
            )


if __name__ == "__main__":
    unittest.main()
