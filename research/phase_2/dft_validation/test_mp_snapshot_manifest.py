from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from pymatgen.core import Lattice, Structure

from build_mp_snapshot_manifest import build_manifest


class MPSnapshotManifestTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        structures = {
            "mp-1": Structure(Lattice.cubic(3.5), ["Li"], [[0, 0, 0]]),
            "mp-2": Structure(
                Lattice.cubic(4.0), ["Li", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]]
            ),
        }
        raw = root / "mp.json"
        raw.write_text(json.dumps([
            {"material_id": material_id, "structure": "fixture"}
            for material_id in structures
        ]), encoding="utf-8")
        metadata = root / "materials.csv"
        with metadata.open("w", newline="", encoding="utf-8") as handle:
            fields = ["uid", "source_id", "formula", "elements_str"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({
                "uid": "MP_mp-1", "source_id": "mp-1",
                "formula": "Li", "elements_str": "Li",
            })
            writer.writerow({
                "uid": "MP_mp-2", "source_id": "mp-2",
                "formula": "LiF", "elements_str": "Li;F",
            })
        cache = root / "structures.sqlite"
        conn = sqlite3.connect(cache)
        try:
            conn.execute(
                "CREATE TABLE structures (material_id TEXT PRIMARY KEY, cif TEXT, formation_energy REAL)"
            )
            for material_id, structure in structures.items():
                conn.execute(
                    "INSERT INTO structures VALUES (?, ?, ?)",
                    (material_id, structure.to(fmt="cif"), 0.0),
                )
            conn.commit()
        finally:
            conn.close()
        return raw, metadata, cache

    def test_verifies_exact_ids_and_cif_compositions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw, metadata, cache = self._fixture(root)
            result = build_manifest(
                raw_json_path=raw,
                materials_csv_path=metadata,
                structure_cache_path=cache,
                output_path=root / "snapshot.json",
            )
            self.assertEqual(result["material_count"], 2)
            self.assertEqual(
                result["composition_audit_status"],
                "all_cache_cifs_match_metadata",
            )

    def test_rejects_filtered_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw, metadata, cache = self._fixture(root)
            lines = metadata.read_text(encoding="utf-8").splitlines()
            metadata.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_manifest(
                    raw_json_path=raw,
                    materials_csv_path=metadata,
                    structure_cache_path=cache,
                    output_path=root / "snapshot.json",
                )

    def test_refuses_to_overwrite_existing_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw, metadata, cache = self._fixture(root)
            output = root / "snapshot.json"
            output.write_text("operator-owned", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                build_manifest(
                    raw_json_path=raw,
                    materials_csv_path=metadata,
                    structure_cache_path=cache,
                    output_path=output,
                )
            self.assertEqual(output.read_text(encoding="utf-8"), "operator-owned")


if __name__ == "__main__":
    unittest.main()
