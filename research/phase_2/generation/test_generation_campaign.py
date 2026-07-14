from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from generation_campaign import (
    BASE_MANIFEST_FIELDS,
    aggregate_campaign,
    build_seed_command,
    parse_seeds,
)


def _candidate(
    candidate_id: str,
    formula: str,
    prototype_uid: str,
    energy: float,
    substitutions: list[dict],
    num_atoms: int = 6,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "formula": formula,
        "prototype_uid": prototype_uid,
        "prototype_formula": "prototype",
        "num_atoms": num_atoms,
        "num_edges": 12,
        "gnn_formation_energy": energy,
        "generation_method": "vae_interpolation",
        "latent_alpha": 0.5,
        "oxidation_states": "{}",
        "substitutions_json": json.dumps(substitutions),
        "num_substitutions": len(substitutions),
        "structure_status": "not_built",
        "cif_path": "",
    }


def _write_run(
    campaign_dir: Path,
    seed: int,
    rows: list[dict[str, object]],
    rejected: dict[str, int],
) -> None:
    run_dir = campaign_dir / "runs" / f"seed_{seed}"
    run_dir.mkdir(parents=True)
    with (run_dir / "generation_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASE_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "generation_seed": seed,
        "n_retrieved": 3,
        "retrieved_uids": ["MP_a", "MP_b"],
        "n_generated": len(rows),
        "n_validated": len(rows),
        "n_samples_requested": 5,
        "elapsed_seconds": 1.0,
        "chemistry_filter_summary": {
            "generated": 5,
            "passed": len(rows),
            "rejected": sum(rejected.values()),
            "rejection_counts": rejected,
        },
    }
    (run_dir / "generation_summary.json").write_text(json.dumps(summary), encoding="utf-8")


class GenerationCampaignTests(unittest.TestCase):
    def test_seed_ranges_are_unique_and_inclusive(self):
        self.assertEqual(parse_seeds("42-44,44,50"), [42, 43, 44, 50])

    def test_command_keeps_all_per_seed_candidates_before_global_dedup(self):
        config = {
            "requirement": "test",
            "limit": 5,
            "n_samples_per_seed": 17,
            "per_seed_top_k": 17,
            "perturb_scale": 0.5,
            "vae_checkpoint": {"path": "/tmp/vae.pt"},
            "gnn_checkpoint": {"path": "/tmp/gnn.pt"},
            "max_energy": 0.0,
            "min_energy": None,
            "include_elements": ["W", "C"],
            "only_elements": [],
            "domain_filter": "refractory_carbide_v1",
            "allowed_elements": [],
            "interpolate": True,
        }
        command = build_seed_command(config, seed=42, output_dir=Path("/tmp/out"))
        self.assertEqual(command[command.index("--top-k") + 1], "17")
        self.assertEqual(command[command.index("--seed") + 1], "42")

    def test_exact_dedup_retains_different_prototype_hypotheses(self):
        substitution = [{"site_index": 0, "from_Z": 73, "to_Z": 22}]
        same_substitution_different_key_order = [
            {"to_Z": 22, "from_Z": 73, "site_index": 0}
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            campaign_dir = Path(temp_dir)
            _write_run(
                campaign_dir,
                1,
                [
                    _candidate("uuid_a", "Ti2WC3", "MP_proto_a", -0.40, substitution),
                    _candidate("uuid_b", "NbWC2", "MP_proto_c", -0.20, substitution, 4),
                ],
                {"known_composition": 3},
            )
            _write_run(
                campaign_dir,
                2,
                [
                    _candidate(
                        "uuid_c", "Ti2WC3", "MP_proto_a", -0.41,
                        same_substitution_different_key_order,
                    ),
                    _candidate("uuid_d", "Ti2WC3", "MP_proto_b", -0.39, substitution),
                ],
                {"unchanged_from_prototype": 3},
            )
            config = {
                "schema_version": "generation_campaign_v1",
                "seeds": [1, 2],
                "global_top_k": 10,
            }
            summary = aggregate_campaign(campaign_dir, config=config)

            self.assertEqual(summary["total_latent_samples_decoded"], 10)
            self.assertEqual(summary["emitted_manifest_occurrence_count"], 4)
            self.assertEqual(summary["unique_pre_cif_hypothesis_count"], 3)
            self.assertEqual(summary["unique_reduced_formula_count"], 2)
            self.assertEqual(summary["exact_duplicate_occurrences_removed"], 1)
            self.assertEqual(summary["same_formula_structural_variants_retained"], 1)
            self.assertEqual(
                summary["chemistry_rejection_counts"],
                {"known_composition": 3, "unchanged_from_prototype": 3},
            )

            with (campaign_dir / "campaign_generation_manifest.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            ti_rows = [row for row in rows if row["formula"] == "Ti2WC3"]
            self.assertEqual(len(ti_rows), 2)
            repeated = next(row for row in ti_rows if row["prototype_uid"] == "MP_proto_a")
            self.assertEqual(repeated["occurrence_count"], "2")
            self.assertEqual(repeated["source_seed"], "2")
            self.assertAlmostEqual(float(repeated["gnn_formation_energy"]), -0.41)
            self.assertTrue(repeated["candidate_id"].startswith("camp_"))

    def test_no_completed_seed_is_a_failed_campaign(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            campaign_dir = Path(temp_dir)
            config = {
                "schema_version": "generation_campaign_v1",
                "seeds": [1, 2],
                "global_top_k": 10,
            }
            with self.assertRaisesRegex(RuntimeError, "No seed run completed"):
                aggregate_campaign(campaign_dir, config=config)
            summary = json.loads(
                (campaign_dir / "campaign_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["campaign_status"], "failed")
            self.assertEqual(summary["failed_seeds"], [1, 2])


if __name__ == "__main__":
    unittest.main()
