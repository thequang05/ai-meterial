from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pymatgen.core import Lattice, Structure

from prepare_qe_jobs import prepare_jobs
from prepare_sssp_manifest import create_manifest


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrepareQEJobsTests(unittest.TestCase):
    def _fixtures(self, root: Path) -> tuple[Path, Path]:
        first = Structure(
            Lattice.cubic(4.3), ["Ti", "C"], [[0, 0, 0], [0.5, 0.5, 0.5]]
        )
        second = Structure(
            Lattice.cubic(4.7), ["Zr", "C"], [[0, 0, 0], [0.5, 0.5, 0.5]]
        )
        first_path = root / "tic.cif"
        second_path = root / "zrc.cif"
        first.to(filename=first_path)
        second.to(filename=second_path)
        report_path = root / "report.json"
        _write_json(report_path, {
            "status": "ml_structural_validation_complete",
            "recommended_dft_queue": [
                {
                    "rank": 1,
                    "candidate_id": "tic",
                    "formula": "TiC",
                    "relaxed_cif": str(first_path),
                    "relaxed_cif_sha256": _hash(first_path),
                },
                {
                    "rank": 2,
                    "candidate_id": "zrc",
                    "formula": "ZrC",
                    "relaxed_cif": str(second_path),
                    "relaxed_cif_sha256": _hash(second_path),
                },
            ],
        })
        config_path = root / "config.json"
        _write_json(config_path, {
            "schema_version": "qe_dft_config_v1",
            "input_dft": "PBE",
            "kpoint_spacing_inv_angstrom": 0.3,
            "occupations": "smearing",
            "smearing": "mv",
            "degauss_ry": 0.02,
            "conv_thr": 1e-8,
            "electron_maxstep": 100,
            "mixing_beta": 0.3,
            "diagonalization": "david",
            "nstep": 100,
            "etot_conv_thr_ry": 1e-5,
            "forc_conv_thr_ry_per_bohr": 8e-4,
            "press_conv_thr_kbar": 0.5,
            "cell_dofree": "all",
            "resource_policy": {"max_parallel_jobs": 1},
        })
        return report_path, config_path

    def test_plan_only_is_blocked_and_does_not_emit_qe_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report, config = self._fixtures(root)
            output = root / "out"

            result = prepare_jobs(
                validation_report_path=report,
                config_path=config,
                output_dir=output,
                top_n=2,
                pw_executable="definitely_missing_pw.x",
            )

            self.assertEqual(result["status"], "blocked_missing_pseudopotentials")
            self.assertFalse(result["calculation_started"])
            self.assertEqual(result["dft_validated_count"], 0)
            self.assertEqual(result["required_elements"], ["C", "Ti", "Zr"])
            self.assertFalse(any(output.glob("jobs/*/vc-relax.in")))
            with (output / "dft_queue_manifest.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["job_status"] == "planned_waiting_for_pseudopotentials" for row in rows))
            self.assertTrue(all(row["qe_scratch_outdir"] == "./tmp" for row in rows))
            self.assertIsNone(result["scratch_root"])

    def test_tracked_wc_candidate_bundle_is_complete_and_hash_verified(self):
        phase2 = Path(__file__).resolve().parents[1]
        report = phase2 / "reports/w_c_structural_validation_v1.json"
        config = Path(__file__).resolve().parent / "qe_dft_config_v1.json"
        with tempfile.TemporaryDirectory() as temp_dir:
            result = prepare_jobs(
                validation_report_path=report,
                config_path=config,
                output_dir=Path(temp_dir) / "source_queue",
                top_n=5,
                convergence_source_only=True,
            )

        self.assertEqual(result["candidate_count"], 5)
        self.assertEqual(
            result["required_elements"], ["C", "Ti", "V", "Zr", "Nb", "Ta", "W"]
        )

    def test_report_relative_cif_paths_resolve_from_report_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report, config = self._fixtures(root)
            payload = json.loads(report.read_text(encoding="utf-8"))
            for row in payload["recommended_dft_queue"]:
                row["relaxed_cif"] = Path(row["relaxed_cif"]).name
            report.write_text(json.dumps(payload), encoding="utf-8")

            result = prepare_jobs(
                validation_report_path=report,
                config_path=config,
                output_dir=root / "relative_report_campaign",
                top_n=2,
            )

            self.assertEqual(result["candidate_count"], 2)
            self.assertEqual(result["required_elements"], ["C", "Ti", "Zr"])

    def test_relaxed_cif_hash_mismatch_is_rejected_before_input_generation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report, config = self._fixtures(root)
            payload = json.loads(report.read_text(encoding="utf-8"))
            first_cif = Path(payload["recommended_dft_queue"][0]["relaxed_cif"])
            first_cif.write_text(first_cif.read_text() + "# tampered\n")

            with self.assertRaisesRegex(ValueError, "CIF SHA-256 mismatch"):
                prepare_jobs(
                    validation_report_path=report,
                    config_path=config,
                    output_dir=root / "tampered_campaign",
                    top_n=1,
                )

    def test_explicit_scratch_is_campaign_isolated_and_audited(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report, config = self._fixtures(root)
            scratch_root = root / "qe_scratch"
            prepared: list[tuple[dict, dict[str, str]]] = []
            for campaign_name in ("campaign_a", "campaign_b"):
                result = prepare_jobs(
                    validation_report_path=report,
                    config_path=config,
                    output_dir=root / campaign_name,
                    top_n=1,
                    pw_executable="definitely_missing_pw.x",
                    scratch_root=scratch_root,
                )
                with Path(result["queue_manifest"]).open(
                    newline="", encoding="utf-8"
                ) as handle:
                    prepared.append((result, next(csv.DictReader(handle))))

            first_result, first_row = prepared[0]
            second_result, second_row = prepared[1]
            first_outdir = Path(first_row["qe_scratch_outdir"])
            second_outdir = Path(second_row["qe_scratch_outdir"])
            self.assertNotEqual(first_outdir, second_outdir)
            self.assertTrue(first_outdir.is_dir())
            self.assertTrue(second_outdir.is_dir())
            self.assertEqual(Path(first_result["scratch_root"]), scratch_root.resolve())
            self.assertNotEqual(
                first_result["scratch_campaign_namespace"],
                second_result["scratch_campaign_namespace"],
            )
            self.assertTrue(first_outdir.is_relative_to(scratch_root.resolve()))
            self.assertTrue(second_outdir.is_relative_to(scratch_root.resolve()))

    def test_nonempty_output_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report, config = self._fixtures(root)
            output = root / "partial_campaign"
            output.mkdir()
            (output / "stale.txt").write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "non-empty output directory"):
                prepare_jobs(
                    validation_report_path=report,
                    config_path=config,
                    output_dir=output,
                    top_n=1,
                )

    def test_verified_pseudos_require_convergence_certificate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report, config = self._fixtures(root)
            pseudo_dir = root / "pseudos"
            pseudo_dir.mkdir()
            official_metadata: dict[str, dict] = {}
            cutoff = {"C": (40, 320), "Ti": (50, 400), "Zr": (60, 480)}
            for symbol, (ecutwfc, ecutrho) in cutoff.items():
                pseudo = pseudo_dir / f"{symbol}.pbe.test.UPF"
                pseudo.write_text(
                    f'<UPF><PP_HEADER element="{symbol}" functional="PBE" '
                    'relativistic="scalar"/></UPF>',
                    encoding="utf-8",
                )
                official_metadata[symbol] = {
                    "filename": pseudo.name,
                    "md5": hashlib.md5(pseudo.read_bytes()).hexdigest(),
                    "cutoff": ecutwfc,
                    "dual": ecutrho / ecutwfc,
                    "pseudopotential": "synthetic-test-family",
                }
            official_metadata_path = root / "official_metadata.json"
            _write_json(official_metadata_path, official_metadata)
            pseudo_manifest = root / "pseudo_manifest.json"
            create_manifest(
                metadata_path=official_metadata_path,
                pseudo_dir=pseudo_dir,
                output_path=pseudo_manifest,
                library_name="SSSP PBE Precision",
                library_version="1",
                elements=list(cutoff),
                acknowledge_original_licenses=True,
            )
            output = root / "out"

            result = prepare_jobs(
                validation_report_path=report,
                config_path=config,
                output_dir=output,
                top_n=2,
                pseudo_manifest_path=pseudo_manifest,
                pseudo_dir=pseudo_dir,
                pw_executable="/usr/bin/true",
            )

            self.assertEqual(
                result["status"], "blocked_missing_convergence_certificate"
            )
            self.assertEqual(result["global_ecutwfc_ry"], 60)
            self.assertEqual(result["global_ecutrho_ry"], 480)
            self.assertFalse(result["calculation_started"])
            inputs = sorted(output.glob("jobs/*/vc-relax.in"))
            self.assertEqual(inputs, [])
            self.assertFalse(result["production_settings_certified"])
            self.assertEqual(len(list((output / "pseudos").glob("*.UPF"))), 3)

    def test_handwritten_manifest_cannot_unlock_qe_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report, config = self._fixtures(root)
            pseudo_dir = root / "pseudos"
            pseudo_dir.mkdir()
            entries = {}
            for symbol in ("C", "Ti", "Zr"):
                pseudo = pseudo_dir / f"{symbol}.UPF"
                pseudo.write_text(
                    f'<UPF><PP_HEADER element="{symbol}" functional="PBE" '
                    'relativistic="scalar"/></UPF>',
                    encoding="utf-8",
                )
                entries[symbol] = {
                    "filename": pseudo.name,
                    "sha256": _hash(pseudo),
                    "ecutwfc_ry": 50,
                    "ecutrho_ry": 400,
                }
            manifest = root / "handwritten.json"
            _write_json(manifest, {
                "schema_version": "qe_pseudo_manifest_v1",
                "library": "self-declared",
                "library_version": "1",
                "functional": "PBE",
                "relativistic": "scalar_relativistic",
                "elements": entries,
            })
            output = root / "blocked"
            result = prepare_jobs(
                validation_report_path=report,
                config_path=config,
                output_dir=output,
                top_n=2,
                pseudo_manifest_path=manifest,
                pseudo_dir=pseudo_dir,
                pw_executable="/usr/bin/true",
            )
            self.assertEqual(result["status"], "blocked_missing_pseudopotentials")
            self.assertFalse(any(output.glob("jobs/*/vc-relax.in")))
            self.assertIn(
                "pseudo_original_licenses_not_acknowledged",
                result["pseudo_blockers"],
            )


if __name__ == "__main__":
    unittest.main()
