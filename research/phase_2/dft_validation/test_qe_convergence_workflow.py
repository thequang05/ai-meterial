from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pymatgen.core import Lattice, Structure

from collect_qe_convergence import (
    RY_TO_EV,
    _parse_force_components,
    _select_stable_tail,
    collect_convergence,
)
from prepare_qe_convergence_jobs import (
    prepare_convergence_jobs,
    validate_protocol,
)
from prepare_qe_jobs import prepare_jobs
from prepare_sssp_manifest import create_manifest
from run_qe_jobs import run_jobs


PROTOCOL_PATH = Path(__file__).with_name("qe_convergence_protocol_v1.json")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_queue(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _build_source_preflight(root: Path) -> tuple[Path, dict]:
    structures = {
        "tic_candidate": Structure(
            Lattice.cubic(4.3), ["Ti", "C"], [[0, 0, 0], [0.5, 0.5, 0.5]]
        ),
        "zrc_candidate": Structure(
            Lattice.cubic(4.7), ["Zr", "C"], [[0, 0, 0], [0.5, 0.5, 0.5]]
        ),
    }
    queue = []
    for rank, (candidate_id, structure) in enumerate(structures.items(), start=1):
        cif = root / f"{candidate_id}.cif"
        structure.to(filename=cif)
        queue.append({
            "rank": rank,
            "candidate_id": candidate_id,
            "formula": structure.composition.reduced_formula,
            "relaxed_cif": str(cif),
        })
    report = root / "report.json"
    _write_json(report, {
        "status": "ml_structural_validation_complete",
        "recommended_dft_queue": queue,
    })
    config = root / "config.json"
    _write_json(config, {
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
        "static_kpoint_spacing_inv_angstrom": 0.2,
        "static_degauss_ry": 0.01,
        "static_conv_thr": 1e-10,
        "resource_policy": {
            "max_parallel_jobs": 1,
            "recommended_mpi_ranks_on_this_mac": 2,
            "omp_threads_per_rank": 1,
        },
    })
    pseudo_dir = root / "source_pseudos"
    pseudo_dir.mkdir()
    metadata: dict[str, dict] = {}
    for symbol, cutoff in (("C", 50), ("Ti", 55), ("Zr", 60)):
        pseudo = pseudo_dir / f"{symbol}.UPF"
        pseudo.write_text(
            f'<UPF><PP_HEADER element="{symbol}" functional="PBE" '
            'relativistic="scalar"/></UPF>',
            encoding="utf-8",
        )
        metadata[symbol] = {
            "filename": pseudo.name,
            "md5": hashlib.md5(pseudo.read_bytes()).hexdigest(),
            "cutoff": cutoff,
            "dual": 8,
            "pseudopotential": "synthetic-test-family",
        }
    metadata_path = root / "official_pseudo_metadata.json"
    _write_json(metadata_path, metadata)
    pseudo_manifest = root / "pseudo.json"
    create_manifest(
        metadata_path=metadata_path,
        pseudo_dir=pseudo_dir,
        output_path=pseudo_manifest,
        library_name="SSSP PBE Precision",
        library_version="1",
        elements=["C", "Ti", "Zr"],
        acknowledge_original_licenses=True,
    )
    source_dir = root / "source_relax"
    preflight = prepare_jobs(
        validation_report_path=report,
        config_path=config,
        output_dir=source_dir,
        top_n=2,
        pseudo_manifest_path=pseudo_manifest,
        pseudo_dir=pseudo_dir,
        pw_executable="/usr/bin/true",
    )
    return source_dir / "dft_preflight.json", preflight


def _prepare_convergence(root: Path) -> tuple[Path, dict]:
    source_preflight, _ = _build_source_preflight(root)
    output_dir = root / "convergence"
    preflight = prepare_convergence_jobs(
        source_preflight_path=source_preflight,
        representative_ids=["tic_candidate", "zrc_candidate"],
        protocol_path=PROTOCOL_PATH,
        output_dir=output_dir,
        pw_executable="/usr/bin/true",
    )
    return output_dir / "convergence_preflight.json", preflight


def _write_synthetic_outputs(preflight_path: Path, preflight: dict) -> None:
    rows = _read_queue(Path(preflight["queue_manifest"]))
    executable_sha = _sha(Path("/usr/bin/true"))
    preflight_sha = _sha(preflight_path)
    for row in rows:
        num_atoms = int(float(row["num_atoms"]))
        rep_offset = -0.1 if row["representative_id"] == "zrc_candidate" else 0.0
        delta_ev_per_atom = 0.0 if row["sweep_axis"] == "anchor" else 0.001
        energy_ev_per_atom = -10.0 + rep_offset + delta_ev_per_atom
        total_energy_ry = energy_ev_per_atom * num_atoms / RY_TO_EV
        force = 0.0 if row["sweep_axis"] == "anchor" else 0.0001
        stress = 0.0 if row["sweep_axis"] == "anchor" else 0.1
        force_lines = "\n".join(
            f"atom {index + 1:4d} type 1   force = {force:.8f} 0.00000000 0.00000000"
            for index in range(num_atoms)
        )
        output = Path(row["qe_output"])
        output.write_text(
            "\n".join([
                "Program PWSCF v.7.4 starts",
                "convergence has been achieved in 8 iterations",
                f"!    total energy = {total_energy_ry:.12f} Ry",
                "Forces acting on atoms (cartesian axes, Ry/au):",
                force_lines,
                "Total force = 0.000100 Ry/Bohr",
                "total   stress  (Ry/bohr**3) (kbar) P= 0.00",
                f"0.0 0.0 0.0  {stress:.8f} 0.0 0.0",
                f"0.0 0.0 0.0  0.0 {stress:.8f} 0.0",
                f"0.0 0.0 0.0  0.0 0.0 {stress:.8f}",
                "JOB DONE.",
                "",
            ]),
            encoding="utf-8",
        )
        Path(row["run_record"]).write_text(json.dumps({
            "run_status": "completed_requires_collection",
            "entry_id": row["entry_id"],
            "entry_role": row["entry_role"],
            "source_id": row["source_id"],
            "candidate_id": row["candidate_id"],
            "formula": row["formula"],
            "qe_input_sha256": row["qe_input_sha256"],
            "qe_output_sha256": _sha(output),
            "preflight_sha256": preflight_sha,
            "queue_manifest_sha256": preflight["queue_manifest_sha256"],
            "preflight_settings_hash": preflight["convergence_settings_hash"],
            "qe_program_version": "7.4",
            "pw_executable_sha256": executable_sha,
        }), encoding="utf-8")


class QeConvergenceWorkflowTests(unittest.TestCase):
    def test_sparse_plan_is_static_and_does_not_execute(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            preflight_path, preflight = _prepare_convergence(root)
            self.assertTrue(preflight["confirmation_eligible"])
            self.assertFalse(preflight["certificate_eligible"])
            rows = _read_queue(Path(preflight["queue_manifest"]))
            self.assertEqual(len(rows), 16)
            self.assertEqual(len({row["point_id"] for row in rows}), 16)
            self.assertEqual(
                sum(row["sweep_axis"] == "anchor" for row in rows), 2
            )
            for row in rows:
                text = Path(row["qe_input"]).read_text(encoding="utf-8")
                self.assertIn("calculation = 'scf'", text)
                self.assertIn("tstress = .TRUE.", text)
                self.assertIn("tprnfor = .TRUE.", text)
                self.assertNotIn("&IONS", text.upper())
                self.assertNotIn("&CELL", text.upper())
                self.assertNotIn("vc-relax", text)
            plan = run_jobs(
                preflight_path=preflight_path,
                execute=False,
                max_jobs=1,
                mpi_ranks=2,
                omp_threads=1,
                mpi_executable="definitely_not_installed_mpirun",
            )
            self.assertEqual(plan["status"], "planned_not_executed")
            self.assertEqual(plan["provenance_blockers"], [])

    def test_collects_provisional_worst_case_selection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            preflight_path, preflight = _prepare_convergence(root)
            _write_synthetic_outputs(preflight_path, preflight)
            summary = collect_convergence(
                preflight_path=preflight_path,
                output_dir=root / "collected",
            )
            self.assertEqual(summary["status"], "provisional_selection_ready")
            self.assertTrue(summary["confirmation_eligible"])
            self.assertFalse(summary["certificate_eligible"])
            self.assertTrue(summary["confirmation_required"])
            self.assertTrue(summary["cutoff_window_converged"])
            self.assertTrue(summary["kpoint_window_converged"])
            self.assertAlmostEqual(
                summary["provisional_global_selection"]["cutoff_pair_multiplier"],
                1.0,
            )
            self.assertAlmostEqual(
                summary["provisional_global_selection"][
                    "kpoint_spacing_inv_angstrom"
                ],
                0.3,
            )

    def test_missing_force_block_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            preflight_path, preflight = _prepare_convergence(root)
            _write_synthetic_outputs(preflight_path, preflight)
            row = _read_queue(Path(preflight["queue_manifest"]))[0]
            output = Path(row["qe_output"])
            text = output.read_text(encoding="utf-8")
            text = text.replace("Forces acting on atoms", "Forces unavailable for atoms")
            output.write_text(text, encoding="utf-8")
            record_path = Path(row["run_record"])
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["qe_output_sha256"] = _sha(output)
            record_path.write_text(json.dumps(record), encoding="utf-8")
            summary = collect_convergence(
                preflight_path=preflight_path,
                output_dir=root / "blocked_collection",
            )
            self.assertEqual(summary["status"], "blocked_incomplete_or_failed_runs")
            self.assertFalse(summary["confirmation_eligible"])

    def test_partial_representative_coverage_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_preflight, _ = _build_source_preflight(root)
            with self.assertRaisesRegex(ValueError, "do not cover"):
                prepare_convergence_jobs(
                    source_preflight_path=source_preflight,
                    representative_ids=["tic_candidate"],
                    protocol_path=PROTOCOL_PATH,
                    output_dir=root / "partial",
                    pw_executable="/usr/bin/true",
                )

    def test_protocol_validation_rejects_unsafe_values(self):
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        protocol["cutoff_pair_multipliers"][0] = 0.9
        with self.assertRaises(ValueError):
            validate_protocol(protocol)

    def test_protocol_snapshot_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            preflight_path, preflight = _prepare_convergence(root)
            protocol_path = Path(preflight["protocol"])
            protocol_path.write_text(
                protocol_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                collect_convergence(
                    preflight_path=preflight_path,
                    output_dir=root / "tampered_collection",
                )

    def test_pseudopotential_tamper_blocks_collection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            preflight_path, preflight = _prepare_convergence(root)
            _write_synthetic_outputs(preflight_path, preflight)
            pseudo = Path(preflight["bundled_pseudopotentials"][0]["path"])
            pseudo.write_text(
                pseudo.read_text(encoding="utf-8") + "tampered",
                encoding="utf-8",
            )
            summary = collect_convergence(
                preflight_path=preflight_path,
                output_dir=root / "tampered_pseudo_collection",
            )
            self.assertEqual(summary["status"], "blocked_incomplete_or_failed_runs")

    def test_stable_tail_does_not_accept_an_earlier_oscillation(self):
        rows = [
            {"point_within_tolerances": True, "level": 0},
            {"point_within_tolerances": False, "level": 1},
            {"point_within_tolerances": True, "level": 2},
            {"point_within_tolerances": True, "level": 3},
        ]
        self.assertIs(_select_stable_tail(rows, min_tail=2), rows[2])
        self.assertIsNone(_select_stable_tail(rows[:3], min_tail=2))
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        protocol["min_stable_tail_points"] = 2.5
        with self.assertRaises(ValueError):
            validate_protocol(protocol)
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        protocol["force_component_tolerance_ev_per_angstrom"] = True
        with self.assertRaises(ValueError):
            validate_protocol(protocol)

    def test_force_parser_rejects_partial_final_block(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "partial.out"
            output.write_text(
                "\n".join([
                    "Forces acting on atoms (Ry/au):",
                    "atom 1 type 1 force = 0 0 0",
                    "atom 2 type 1 force = 0 0 0",
                    "Forces acting on atoms (Ry/au):",
                    "atom 1 type 1 force = 0 0 0",
                    "JOB DONE.",
                ]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Expected 2"):
                _parse_force_components(output, 2)


if __name__ == "__main__":
    unittest.main()
