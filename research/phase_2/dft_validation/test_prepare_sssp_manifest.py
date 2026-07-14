from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from prepare_sssp_manifest import create_manifest


class PrepareSSSPManifestTests(unittest.TestCase):
    def test_verifies_md5_and_derives_density_cutoff(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pseudo_dir = root / "pseudos"
            pseudo_dir.mkdir()
            pseudo = pseudo_dir / "C.test.UPF"
            pseudo.write_text(
                '<UPF version="2.0.1"><PP_HEADER element="C" '
                'functional="PBE" relativistic="scalar"/></UPF>',
                encoding="utf-8",
            )
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({
                "C": {
                    "filename": pseudo.name,
                    "md5": hashlib.md5(pseudo.read_bytes()).hexdigest(),
                    "cutoff": 45,
                    "dual": 8,
                    "pseudopotential": "test-family",
                }
            }), encoding="utf-8")

            result = create_manifest(
                metadata_path=metadata,
                pseudo_dir=pseudo_dir,
                output_path=root / "manifest.json",
                library_name="test",
                library_version="1",
                elements=["C"],
                acknowledge_original_licenses=True,
            )

            self.assertEqual(result["elements"]["C"]["ecutwfc_ry"], 45)
            self.assertEqual(result["elements"]["C"]["ecutrho_ry"], 360)
            self.assertEqual(len(result["elements"]["C"]["sha256"]), 64)
            self.assertEqual(result["elements"]["C"]["upf_header_functional"], "PBE")

    def test_rejects_missing_official_md5(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pseudo_dir = root / "pseudos"
            pseudo_dir.mkdir()
            pseudo = pseudo_dir / "C.test.UPF"
            pseudo.write_text(
                '<UPF><PP_HEADER element="C" functional="PBE" '
                'relativistic="scalar"/></UPF>',
                encoding="utf-8",
            )
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({
                "C": {
                    "filename": pseudo.name,
                    "cutoff": 45,
                    "dual": 8,
                }
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "MD5"):
                create_manifest(
                    metadata_path=metadata,
                    pseudo_dir=pseudo_dir,
                    output_path=root / "manifest.json",
                    library_name="test",
                    library_version="1",
                    elements=["C"],
                    acknowledge_original_licenses=True,
                )

    def test_requires_explicit_license_acknowledgement(self):
        with self.assertRaisesRegex(ValueError, "acknowledge"):
            create_manifest(
                metadata_path=Path("missing"),
                pseudo_dir=Path("missing"),
                output_path=Path("missing"),
                library_name="test",
                library_version="1",
                elements=["C"],
                acknowledge_original_licenses=False,
            )

    def test_rejects_pbesol_header_in_plain_pbe_workflow(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pseudo_dir = root / "pseudos"
            pseudo_dir.mkdir()
            pseudo = pseudo_dir / "C.pbesol.UPF"
            pseudo.write_text(
                '<UPF><PP_HEADER element="C" functional="PBEsol" '
                'relativistic="scalar"/></UPF>',
                encoding="utf-8",
            )
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({
                "C": {
                    "filename": pseudo.name,
                    "md5": hashlib.md5(pseudo.read_bytes()).hexdigest(),
                    "cutoff": 45,
                    "dual": 8,
                }
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not explicitly PBE"):
                create_manifest(
                    metadata_path=metadata,
                    pseudo_dir=pseudo_dir,
                    output_path=root / "manifest.json",
                    library_name="test",
                    library_version="1",
                    elements=["C"],
                    acknowledge_original_licenses=True,
                )


if __name__ == "__main__":
    unittest.main()
