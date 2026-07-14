"""Create the locked UPF manifest consumed by ``prepare_qe_jobs.py``.

The input is an official SSSP metadata JSON plus the matching extracted UPF
directory.  This command never downloads files and requires an explicit flag
confirming that the user reviewed the original pseudopotential licenses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


OUTPUT_SCHEMA = "qe_pseudo_manifest_v1"
DEFAULT_ELEMENTS = ["C", "Ti", "V", "Zr", "Nb", "Ta", "W"]
PBE_FUNCTIONAL_HEADER_ALIASES = {
    "PBE",
    "PBEGGA",
    "GGAPBE",
    "PERDEWBURKEERNZERHOF",
    # Quantum ESPRESSO component labels for LDA exchange/correlation plus
    # the PBE gradient corrections (the spelling used by some UPF writers).
    "SLAPWPBXPBC",
}


def _digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _normalize_expected_md5(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.removeprefix("md5:")


def _inspect_upf_header(path: Path, expected_symbol: str) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")[:250_000]
    attributes: dict[str, str] = {}
    for name in ("element", "functional", "relativistic"):
        match = re.search(
            rf"\b{name}\s*=\s*['\"]\s*([^'\"]+?)\s*['\"]", text, re.I
        )
        if not match:
            raise ValueError(f"UPF header has no parseable {name}: {path.name}")
        attributes[name] = match.group(1).strip()
    if attributes["element"].title() != expected_symbol:
        raise ValueError(
            f"UPF element mismatch for {expected_symbol}: "
            f"header={attributes['element']!r}, file={path.name}"
        )
    functional = re.sub(r"[^A-Z0-9]", "", attributes["functional"].upper())
    if functional not in PBE_FUNCTIONAL_HEADER_ALIASES:
        raise ValueError(
            f"UPF for {expected_symbol} is not explicitly PBE: "
            f"functional={attributes['functional']!r}"
        )
    relativistic = re.sub(
        r"[^a-z]", "", attributes["relativistic"].lower()
    )
    if relativistic not in {"scalar", "scalarrelativistic"}:
        raise ValueError(
            f"UPF for {expected_symbol} is not scalar-relativistic: "
            f"relativistic={attributes['relativistic']!r}"
        )
    return attributes


def create_manifest(
    *,
    metadata_path: Path,
    pseudo_dir: Path,
    output_path: Path,
    library_name: str,
    library_version: str,
    elements: list[str],
    acknowledge_original_licenses: bool,
) -> dict[str, Any]:
    if not acknowledge_original_licenses:
        raise ValueError(
            "Refusing to create a runnable manifest until "
            "--acknowledge-original-licenses is supplied"
        )
    metadata_path = Path(metadata_path).resolve()
    pseudo_dir = Path(pseudo_dir).resolve()
    output_path = Path(output_path).resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not pseudo_dir.is_dir():
        raise FileNotFoundError(pseudo_dir)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if len(elements) != len(set(elements)):
        raise ValueError("Duplicate element symbols are not allowed")

    selected: dict[str, dict[str, Any]] = {}
    selected_filenames: set[str] = set()
    for symbol in elements:
        entry = metadata.get(symbol)
        if not isinstance(entry, dict):
            raise ValueError(f"SSSP metadata does not contain element {symbol}")
        filename = str(entry.get("filename") or "").strip()
        if not filename:
            raise ValueError(f"SSSP metadata has no filename for {symbol}")
        filename_path = Path(filename)
        if filename_path.is_absolute() or filename_path.parent != Path("."):
            raise ValueError(f"Unsafe pseudopotential filename for {symbol}: {filename}")
        if filename in selected_filenames:
            raise ValueError(f"Duplicate pseudopotential filename: {filename}")
        selected_filenames.add(filename)
        source = pseudo_dir / filename
        if not source.is_file():
            raise FileNotFoundError(source)

        cutoff = float(entry["cutoff"])
        if "dual" in entry:
            dual = float(entry["dual"])
            ecutrho = cutoff * dual
        elif "ecutrho" in entry:
            ecutrho = float(entry["ecutrho"])
            dual = ecutrho / cutoff
        else:
            raise ValueError(f"SSSP metadata has neither dual nor ecutrho for {symbol}")
        if (
            not all(math.isfinite(value) for value in (cutoff, dual, ecutrho))
            or cutoff <= 0
            or dual < 1
            or ecutrho < cutoff
        ):
            raise ValueError(f"Invalid SSSP cutoff metadata for {symbol}")

        expected_md5 = _normalize_expected_md5(entry.get("md5") or entry.get("checksum"))
        if not re.fullmatch(r"[0-9a-f]{32}", expected_md5):
            raise ValueError(f"Official 32-hex MD5 is missing for {symbol}")
        actual_md5 = _digest(source, "md5")
        if actual_md5 != expected_md5:
            raise ValueError(
                f"MD5 mismatch for {symbol}: expected={expected_md5}, actual={actual_md5}"
            )
        upf_header = _inspect_upf_header(source, symbol)
        selected[symbol] = {
            "filename": filename,
            "sha256": _digest(source, "sha256"),
            "original_md5": actual_md5,
            "ecutwfc_ry": cutoff,
            "ecutrho_ry": ecutrho,
            "dual": dual,
            "pseudopotential_family": entry.get("pseudopotential", ""),
            "upf_header_element": upf_header["element"],
            "upf_header_functional": upf_header["functional"],
            "upf_header_relativistic": upf_header["relativistic"],
        }

    manifest = {
        "schema_version": OUTPUT_SCHEMA,
        "library": library_name,
        "library_version": library_version,
        "functional": "PBE",
        "relativistic": "scalar_relativistic",
        "source_metadata": str(metadata_path),
        "source_metadata_sha256": _digest(metadata_path, "sha256"),
        "pseudo_dir_at_manifest_creation": str(pseudo_dir),
        "licenses_acknowledged_by_user": True,
        "license_note": (
            "Each SSSP pseudopotential retains its original license; preserve "
            "the release acknowledgements and cite the individual sources."
        ),
        "elements": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Verified {len(selected)} pseudopotentials: {','.join(selected)}")
    print(f"Manifest: {output_path}")
    return manifest


def _elements(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("At least one element is required")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pseudo-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-name", default="SSSP PBE Precision")
    parser.add_argument("--library-version", required=True)
    parser.add_argument("--elements", type=_elements, default=DEFAULT_ELEMENTS)
    parser.add_argument("--acknowledge-original-licenses", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    create_manifest(
        metadata_path=args.metadata,
        pseudo_dir=args.pseudo_dir,
        output_path=args.output,
        library_name=args.library_name,
        library_version=args.library_version,
        elements=args.elements,
        acknowledge_original_licenses=args.acknowledge_original_licenses,
    )
