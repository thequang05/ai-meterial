from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from compute_qe_hull import RY_TO_EV, _subsystem_keys, compute_hulls


SETTINGS_HASH = "a" * 64
MATERIALS_SHA = "b" * 64
CACHE_SHA = "c" * 64
EXECUTABLE_SHA = "d" * 64
RAW_SHA = "e" * 64
ID_SHA = "f" * 64
AUDIT_SHA = "1" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _receipt(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_entries(path: Path, rows: list[dict]) -> None:
    fields = [
        "entry_id", "entry_role", "source_id", "candidate_id", "formula",
        "composition_json", "num_atoms", "total_energy_ry", "total_energy_ev",
        "energy_ev_per_atom", "static_gate_status", "static_settings_hash",
        "source_cif_sha256", "relaxed_cif_sha256", "qe_input_sha256",
        "qe_output_sha256", "pw_executable_sha256", "entry_lineage_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _entry(
    entry_id: str,
    role: str,
    formula: str,
    composition: dict[str, int],
    total_energy_ev: float,
    *,
    source_id: str,
    source_cif_sha256: str,
) -> dict:
    canonical_composition = {
        symbol: float(amount) for symbol, amount in composition.items()
    }
    composition_json = json.dumps(canonical_composition, sort_keys=True)
    num_atoms = float(sum(composition.values()))
    total_energy_ry = total_energy_ev / RY_TO_EV
    relaxed_sha = _digest(f"relaxed:{entry_id}")
    input_sha = _digest(f"input:{entry_id}")
    output_sha = _digest(f"output:{entry_id}")
    lineage = hashlib.sha256(json.dumps({
        "entry_id": entry_id,
        "entry_role": role,
        "source_id": source_id,
        "formula": formula,
        "composition_json": composition_json,
        "num_atoms": num_atoms,
        "source_cif_sha256": source_cif_sha256,
        "relaxed_cif_sha256": relaxed_sha,
        "qe_input_sha256": input_sha,
        "qe_output_sha256": output_sha,
        "static_settings_hash": SETTINGS_HASH,
        "pw_executable_sha256": EXECUTABLE_SHA,
        "total_energy_ry": total_energy_ry,
    }, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "entry_id": entry_id,
        "entry_role": role,
        "source_id": source_id,
        "candidate_id": "candidate_1" if role == "candidate" else "",
        "formula": formula,
        "composition_json": composition_json,
        "num_atoms": num_atoms,
        "total_energy_ry": total_energy_ry,
        "total_energy_ev": total_energy_ry * RY_TO_EV,
        "energy_ev_per_atom": total_energy_ry * RY_TO_EV / num_atoms,
        "static_gate_status": "dft_static_converged",
        "static_settings_hash": SETTINGS_HASH,
        "source_cif_sha256": source_cif_sha256,
        "relaxed_cif_sha256": relaxed_sha,
        "qe_input_sha256": input_sha,
        "qe_output_sha256": output_sha,
        "pw_executable_sha256": EXECUTABLE_SHA,
        "entry_lineage_sha256": lineage,
    }


def _write_summary(path: Path, entries: Path) -> None:
    path.write_text(json.dumps({
        "collector_version": "qe_static_collector_v1",
        "results_manifest_sha256": _sha(entries),
        "energy_records_sha256": _digest(f"records:{entries.name}"),
        "static_settings_hash": SETTINGS_HASH,
    }), encoding="utf-8")


class ComputeQEHullTests(unittest.TestCase):
    def _files(self, root: Path, *, include_binary: bool = True) -> dict[str, Path]:
        candidate_source_sha = _digest("candidate-source")
        reference_specs = [
            ("ref_Li", "mp-li", "Li", "Li", {"Li": 1}, 0.0),
            ("ref_F", "mp-f", "F2", "F", {"F": 1}, 0.0),
            ("ref_LiF", "mp-lif", "LiF", "F-Li", {"Li": 1, "F": 1}, -4.0),
        ]
        reference_source_hashes = {
            entry_id: _digest(f"source:{entry_id}")
            for entry_id, *_ in reference_specs
        }
        candidates = root / "candidates.csv"
        references = root / "references.csv"
        _write_entries(candidates, [
            _entry(
                "candidate:1", "candidate", "Li2F", {"Li": 2, "F": 1}, -3.0,
                source_id="candidate_1", source_cif_sha256=candidate_source_sha,
            )
        ])
        _write_entries(references, [
            _entry(
                entry_id, "reference", formula, composition, energy,
                source_id=source_id,
                source_cif_sha256=reference_source_hashes[entry_id],
            )
            for entry_id, source_id, formula, _subsystem, composition, energy
            in reference_specs
        ])
        candidate_summary = root / "candidate_summary.json"
        reference_summary = root / "reference_summary.json"
        _write_summary(candidate_summary, candidates)
        _write_summary(reference_summary, references)

        inventory = root / "reference_inventory.csv"
        inventory_fields = [
            "entry_id", "entry_role", "source_id", "formula", "subsystem",
            "cif_materialized", "source_cif_sha256",
        ]
        inventory_rows = [
            {
                "entry_id": entry_id,
                "entry_role": "reference",
                "source_id": source_id,
                "formula": formula,
                "subsystem": subsystem,
                "cif_materialized": True,
                "source_cif_sha256": reference_source_hashes[entry_id],
            }
            for entry_id, source_id, formula, subsystem, _composition, _energy
            in reference_specs
        ]
        with inventory.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=inventory_fields)
            writer.writeheader()
            writer.writerows(inventory_rows)

        snapshot = root / "snapshot.json"
        snapshot.write_text(json.dumps({
            "schema_version": "mp_snapshot_manifest_v1",
            "status": "verified_exact_id_sets",
            "snapshot_id": "mp.2019.04.01",
            "material_count": 3,
            "material_id_sha256": ID_SHA,
            "composition_audit_sha256": AUDIT_SHA,
            "composition_audit_status": "all_cache_cifs_match_metadata",
            "raw_json": {
                "path": "/synthetic/mp.2019.04.01.json",
                "sha256": RAW_SHA,
                "material_count": 3,
                "material_id_occurrences": 3,
                "material_id_sha256": ID_SHA,
            },
            "materials_csv": {
                "path": "/synthetic/materials.csv",
                "sha256": MATERIALS_SHA,
                "material_count": 3,
                "material_id_sha256": ID_SHA,
            },
            "structure_cache": {
                "path": "/synthetic/structures.sqlite",
                "sha256": CACHE_SHA,
                "material_count": 3,
                "material_id_sha256": ID_SHA,
                "composition_audit_sha256": AUDIT_SHA,
            },
        }), encoding="utf-8")
        selected = sorted([
            {
                "entry_id": row["entry_id"],
                "source_id": row["source_id"],
                "formula": row["formula"],
                "subsystem": row["subsystem"],
                "source_cif_sha256": row["source_cif_sha256"],
            }
            for row in inventory_rows
        ], key=lambda row: row["entry_id"])
        inventory_digest = _receipt({
            "inventory_version": "mp2019_reference_inventory_v1",
            "materials_csv_sha256": MATERIALS_SHA,
            "structure_cache_sha256": CACHE_SHA,
            "selected": selected,
            "diagnostic_max_per_subsystem": None,
        })
        ids_by_subsystem = {
            "F": ["ref_F"],
            "Li": ["ref_Li"],
            "F-Li": ["ref_LiF"],
        }
        subsystem_records = {}
        for subsystem, ids in ids_by_subsystem.items():
            if subsystem == "F-Li" and not include_binary:
                continue
            record = {
                "query_status": "complete",
                "expected_reference_ids": ids,
            }
            record["query_receipt_sha256"] = _receipt({
                "inventory_sha256": inventory_digest,
                "subsystem": subsystem,
                "expected_reference_ids": ids,
                "query_status": "complete",
            })
            subsystem_records[subsystem] = record
        coverage = root / "coverage.json"
        coverage.write_text(json.dumps({
            "schema_version": "dft_reference_coverage_v1",
            "coverage_id": "synthetic_test",
            "static_settings_hash": SETTINGS_HASH,
            "inventory": {
                "source_name": "synthetic MP snapshot fixture",
                "source_snapshot_id": "mp.2019.04.01",
                "inventory_sha256": inventory_digest,
                "query_code_version": "mp2019_reference_inventory_v1",
                "scope": "all structures from every non-empty subsystem",
                "materials_csv_sha256": MATERIALS_SHA,
                "structure_cache_sha256": CACHE_SHA,
                "diagnostic_max_per_subsystem": None,
                "reference_inventory": str(inventory),
                "reference_inventory_sha256": _sha(inventory),
                "snapshot_manifest": str(snapshot),
                "snapshot_manifest_sha256": _sha(snapshot),
            },
            "systems": {
                "F-Li": {
                    "elements": ["F", "Li"],
                    "subsystems": subsystem_records,
                    "deduplications": [],
                    "coverage_complete_assertion": True,
                }
            },
        }), encoding="utf-8")
        return {
            "candidates": candidates,
            "references": references,
            "candidate_summary": candidate_summary,
            "reference_summary": reference_summary,
            "inventory": inventory,
            "snapshot": snapshot,
            "coverage": coverage,
        }

    def _compute(self, root: Path, files: dict[str, Path], **kwargs):
        return compute_hulls(
            candidate_entries_path=files["candidates"],
            candidate_summary_path=files["candidate_summary"],
            reference_entries_path=files["references"],
            reference_summary_path=files["reference_summary"],
            reference_inventory_path=files["inventory"],
            coverage_manifest_path=files["coverage"],
            output_dir=root / "out",
            **kwargs,
        )

    def test_complete_binary_inventory_emits_official_hull_value(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary = self._compute(root, self._files(root))
            self.assertEqual(summary["status"], "complete")
            row = summary["candidates"][0]
            self.assertAlmostEqual(
                row["energy_above_augmented_hull_ev_per_atom"], 1 / 3, places=8
            )
            self.assertFalse(row["within_user_supplied_hull_threshold"])
            self.assertIn("finite-smearing", row["claim_scope"])

    def test_missing_subsystem_blocks_official_value(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary = self._compute(
                root, self._files(root, include_binary=False),
                diagnostic_incomplete=True,
            )
            self.assertEqual(summary["status"], "blocked_incomplete_reference_coverage")
            row = summary["candidates"][0]
            self.assertEqual(row["energy_above_augmented_hull_ev_per_atom"], "")
            self.assertEqual(row["hull_class"], "blocked")
            self.assertFalse(row["within_user_supplied_hull_threshold"])

    def test_non_finite_threshold_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(ValueError, "threshold"):
                self._compute(
                    root, self._files(root),
                    near_hull_threshold_ev_per_atom=float("inf"),
                )

    def test_minimal_self_declared_snapshot_cannot_unlock_hull(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = self._files(root)
            snapshot = json.loads(files["snapshot"].read_text(encoding="utf-8"))
            del snapshot["material_count"]
            files["snapshot"].write_text(json.dumps(snapshot), encoding="utf-8")
            coverage = json.loads(files["coverage"].read_text(encoding="utf-8"))
            coverage["inventory"]["snapshot_manifest_sha256"] = _sha(
                files["snapshot"]
            )
            files["coverage"].write_text(json.dumps(coverage), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "material_count"):
                self._compute(root, files)

    def test_non_finite_phase_diagram_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = self._files(root)
            with patch(
                "pymatgen.analysis.phase_diagram.PhaseDiagram."
                "get_decomp_and_e_above_hull",
                return_value=({}, float("nan")),
            ):
                with self.assertRaisesRegex(ValueError, "Non-finite augmented"):
                    self._compute(root, files)

    def test_protocol_caps_subsystem_explosion(self):
        with self.assertRaisesRegex(ValueError, "at most 6 elements"):
            _subsystem_keys({"A", "B", "C", "D", "E", "F", "G"})


if __name__ == "__main__":
    unittest.main()
