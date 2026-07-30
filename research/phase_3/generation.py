"""
Phase 3 - Generative inverse design (natural-language driven).

Flow:
  1. LM Studio LLM (with the Neo4j MCP) turns a natural-language requirement
     into a list of prototype material uids (it does the retrieval via MCP).
  2. Load each prototype's full structure from a local sqlite cache of the
     raw MEGNet JSON.
  3. pymatgen data-mined substitution -> novel, charge-neutral candidates.
  4. Write candidate CIFs + manifest.csv.

Runs in the existing .conda env; only pymatgen + httpx are needed (no new deps,
no direct Neo4j dependency here - retrieval lives in LM Studio's MCP).

Example:
    python generation.py --requirement "stable oxide with low formation energy"
    python generation.py --uids MP_mp-1173034,MP_mp-1100894   # bypass the LLM
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import warnings
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv

import httpx
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.structure_prediction.substitution_probability import (
    SubstitutionPredictor,
)
from pymatgen.core import Structure
from pymatgen.transformations.standard_transformations import SubstitutionTransformation

# pymatgen's CifWriter warns about non-unique (oxidation-decorated) site labels;
# harmless for us, so silence it.
warnings.filterwarnings("ignore", message="Site labels are not unique")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
_PHASE_3 = Path(__file__).resolve().parent
# The raw Materials Project snapshot is stored in the repository's data
# directory.  Older notebooks used dataset/materials_project/; keep that
# location as a backwards-compatible fallback for older checkouts.
_RAW_JSON_PRIMARY = _PROJECT_ROOT / "data" / "mp.2019.04.01.json"
_RAW_JSON_LEGACY = _PROJECT_ROOT / "dataset" / "materials_project" / "mp.2019.04.01.json"
RAW_JSON = _RAW_JSON_PRIMARY if _RAW_JSON_PRIMARY.exists() else _RAW_JSON_LEGACY
CACHE_DB = _PHASE_3 / "cache" / "structures.sqlite"
DEFAULT_OUTPUT = _PHASE_3 / "output"
DEFAULT_LMSTUDIO_URL = "http://192.168.1.47:1234"
DEFAULT_MCP_PLUGIN_ID = "mcp/thanhhoa-materials"

# Matches material ids in either Neo4j ("MP_mp-123") or raw ("mp-123") form.
_UID_RE = re.compile(r"MP_mp-\d+|mp-\d+")

MANIFEST_FIELDS = [
    "candidate_id", "prototype_uid", "prototype_formula", "new_formula",
    "substitution_map", "substitution_probability", "prototype_formation_energy",
    "cif_path",
]


# ── Retrieval via LM Studio native /api/v1/chat + Neo4j MCP -> uids ───────────
# NOTE: the OpenAI-compatible /v1/chat/completions endpoint does NOT execute MCP
# tools (it only emits the tool-call as text). The native /api/v1/chat endpoint
# runs MCP servers configured via `integrations` and returns an `output` array.
_INSTRUCTION = (
    "Use the materials database tools to find the most thermodynamically stable "
    "materials (lowest formation_energy_per_atom, eV/atom) matching this "
    "requirement, then reply with ONLY a JSON array of their uid strings "
    '(e.g. ["MP_mp-1173034", "MP_mp-1100894"]).\n\nRequirement: '
)


def _auth_headers(api_token: Optional[str]) -> dict[str, str]:
    token = api_token or os.getenv("LMSTUDIO_API_TOKEN")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _first_model_id(base_url: str, headers: dict, timeout: float) -> str:
    payload = httpx.get(f"{base_url}/api/v1/models", headers=headers, timeout=timeout).json()
    items = payload.get("data") or payload.get("models") or payload
    if not items:
        raise RuntimeError("LM Studio reports no available models.")
    first = items[0]
    return first.get("id") or first.get("key") or first.get("model_key")


def _extract_uids(output: list[dict], n: int) -> list[str]:
    """Pull uids from the final message (JSON array), else from tool-call outputs."""
    messages = [o.get("content", "") for o in output if o.get("type") == "message"]
    for content in reversed(messages):
        match = re.search(r"\[.*\]", content or "", re.DOTALL)
        if match:
            try:
                arr = json.loads(match.group(0))
                uids = [str(u) for u in arr if isinstance(u, str)]
                if uids:
                    return uids[:n]
            except json.JSONDecodeError:
                pass
    # Fallback: scrape uids from executed tool-call outputs.
    found: list[str] = []
    for item in output:
        if item.get("type") == "tool_call":
            for uid in _UID_RE.findall(str(item.get("output", ""))):
                if uid not in found:
                    found.append(uid)
    if found:
        return found[:n]
    raise ValueError(f"No uids found in LM Studio response output: {output!r}")


def get_prototype_uids(
    requirement: str,
    n: int,
    base_url: str = DEFAULT_LMSTUDIO_URL,
    plugin_id: str = DEFAULT_MCP_PLUGIN_ID,
    api_token: Optional[str] = None,
    timeout: float = 300.0,
) -> list[str]:
    """Ask the LM Studio model (with the Neo4j MCP plugin) for prototype uids."""
    headers = _auth_headers(api_token)
    body = {
        "model": _first_model_id(base_url, headers, timeout),
        "input": f"{_INSTRUCTION}{requirement}\n\nReturn up to {n} uids.",
        "integrations": [{"type": "plugin", "id": plugin_id}],
        "context_length": 8000,
    }
    resp = httpx.post(
        f"{base_url}/api/v1/chat", headers=headers, json=body, timeout=timeout
    )
    if resp.status_code in (401, 403):
        raise RuntimeError(
            "LM Studio rejected the request (auth). Local MCP plugins require an "
            "API token: create one in LM Studio (Developer > API tokens) and set "
            "the LMSTUDIO_API_TOKEN environment variable, or pass --uids to bypass."
        )
    resp.raise_for_status()
    return _extract_uids(resp.json().get("output", []), n)


# ── Structure cache (raw JSON -> sqlite -> pymatgen Structure) ────────────────
def build_structure_cache(force: bool = False) -> int:
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_DB)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS structures ("
            "material_id TEXT PRIMARY KEY, cif TEXT, formation_energy REAL)"
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM structures").fetchone()[0]
        if count and not force:
            return count
        if not RAW_JSON.exists():
            raise FileNotFoundError(f"Raw dataset not found at {RAW_JSON}")
        print(f"Building structure cache from {RAW_JSON.name} (one-time, ~1 min) ...")
        with open(RAW_JSON, encoding="utf-8") as fh:
            data = json.load(fh)
        rows = [(e["material_id"], e["structure"],
                 e.get("formation_energy_per_atom")) for e in data]
        conn.execute("DELETE FROM structures")
        conn.executemany("INSERT OR REPLACE INTO structures VALUES (?, ?, ?)", rows)
        conn.commit()
        print(f"  cached {len(rows):,} structures -> {CACHE_DB}")
        return len(rows)
    finally:
        conn.close()


def load_prototype(
    uid: str, conn: sqlite3.Connection
) -> Optional[tuple[Structure, Optional[float]]]:
    """uid 'MP_mp-1103373' -> (Structure, formation_energy) from the cache."""
    material_id = uid[3:] if uid.startswith("MP_") else uid
    row = conn.execute(
        "SELECT cif, formation_energy FROM structures WHERE material_id = ?",
        (material_id,),
    ).fetchone()
    if not row or not row[0]:
        return None
    try:
        return Structure.from_str(row[0], fmt="cif"), row[1]
    except Exception as exc:  # noqa: BLE001
        print(f"  ! failed to parse structure for {uid}: {exc}")
        return None


# ── Substitution generator (pymatgen data-mined model) ───────────────────────
def _format_map(mapping: dict) -> str:
    return "; ".join(f"{k}->{v}" for k, v in mapping.items() if k != v)


def generate_candidates(
    structure: Structure, max_per_prototype: int, prob_threshold: float
) -> list[tuple[Structure, float, dict]]:
    decorated = structure.copy()
    try:
        decorated.add_oxidation_state_by_guess()
    except Exception as exc:  # noqa: BLE001
        print(f"  ! oxidation-state guess failed, skipping: {exc}")
        return []

    species = list(dict.fromkeys(decorated.species))
    try:
        # to_this_composition=False -> substitutions map {prototype_sp: new_sp}
        preds = SubstitutionPredictor(threshold=prob_threshold).list_prediction(
            species, to_this_composition=False
        )
    except ValueError as exc:
        print(f"  ! substitution model rejected prototype species: {exc}")
        return []

    preds.sort(key=lambda d: d["probability"], reverse=True)
    results: list[tuple[Structure, float, dict]] = []
    for pred in preds:
        mapping = pred["substitutions"]
        if all(k == v for k, v in mapping.items()):
            continue  # identity = the prototype itself
        try:
            new_struct = SubstitutionTransformation(mapping).apply_transformation(decorated)
        except Exception:  # noqa: BLE001
            continue
        if abs(sum(getattr(s.specie, "oxi_state", 0.0) for s in new_struct)) > 1e-6:
            continue  # keep only charge-neutral candidates
        out = new_struct.copy()
        out.remove_oxidation_states()
        results.append((out, float(pred["probability"]), dict(mapping)))
        if len(results) >= max_per_prototype:
            break
    return results


# ── Orchestration ────────────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> None:
    cif_dir = Path(args.output) / "cifs"
    cif_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.output) / "manifest.csv"

    cached = build_structure_cache(force=args.rebuild_cache)
    if args.build_cache_only:
        print(f"Structure cache ready: {cached:,} structures -> {CACHE_DB}")
        return
    conn = sqlite3.connect(CACHE_DB)

    # Prototype uids: explicit --uids override, else ask the LLM (+ Neo4j MCP).
    if args.uids:
        uids = args.uids
    else:
        print(f"Asking LM Studio (+ MCP) for prototypes: {args.requirement!r}")
        uids = get_prototype_uids(
            args.requirement,
            args.n_prototypes,
            base_url=args.lmstudio_url,
            plugin_id=args.mcp_plugin_id,
        )
    print(f"  prototypes: {uids}")

    matcher = StructureMatcher()
    kept: list[Structure] = []
    rows: list[dict[str, Any]] = []
    idx = 0

    for uid in uids:
        if idx >= args.max_candidates:
            break
        loaded = load_prototype(uid, conn)
        if loaded is None:
            print(f"  ! no cached structure for {uid}; skipping.")
            continue
        structure, energy = loaded
        proto_formula = structure.composition.reduced_formula
        print(f"Prototype {uid} ({proto_formula}): generating ...")

        for cand, prob, mapping in generate_candidates(
            structure, args.max_per_prototype, args.prob_threshold
        ):
            if idx >= args.max_candidates:
                break
            if cand.composition.reduced_formula == proto_formula:
                continue
            formula = cand.composition.reduced_formula
            if any(formula == k.composition.reduced_formula and matcher.fit(cand, k)
                   for k in kept):
                continue

            cid = f"cand_{idx:04d}"
            cif_path = cif_dir / f"{cid}.cif"
            cand.to(filename=str(cif_path))
            rows.append({
                "candidate_id": cid,
                "prototype_uid": uid,
                "prototype_formula": proto_formula,
                "new_formula": formula,
                "substitution_map": _format_map(mapping),
                "substitution_probability": f"{prob:.6g}",
                "prototype_formation_energy": energy,
                "cif_path": str(cif_path),
            })
            kept.append(cand)
            idx += 1
            print(f"  + {cid}: {formula} (p={prob:.3g}) from {_format_map(mapping)}")

    conn.close()

    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} candidate(s).\n  CIFs:     {cif_dir}"
          f"\n  Manifest: {manifest_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 3 generative inverse design.")
    p.add_argument("--requirement",
                   default="stable material with low formation energy",
                   help="Natural-language design requirement (LLM + Neo4j MCP).")
    p.add_argument("--uids", type=lambda s: [u.strip() for u in s.split(",") if u.strip()],
                   help="Comma-separated prototype uids to bypass the LLM (testing).")
    p.add_argument("--n-prototypes", type=int, default=5,
                   help="How many prototypes to request from the LLM.")
    p.add_argument("--max-per-prototype", type=int, default=5,
                   help="Max candidates generated per prototype.")
    p.add_argument("--max-candidates", type=int, default=10,
                   help="Global cap on candidates.")
    p.add_argument("--prob-threshold", type=float, default=1e-3,
                   help="Substitution-probability pruning threshold.")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help="Output directory for CIFs + manifest.csv.")
    p.add_argument("--lmstudio-url", default=DEFAULT_LMSTUDIO_URL,
                   help="LM Studio base URL (native /api/v1 endpoints).")
    p.add_argument("--mcp-plugin-id", default=DEFAULT_MCP_PLUGIN_ID,
                   help="LM Studio MCP plugin id from mcp.json (e.g. mcp/thanhhoa-materials).")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="Force-rebuild the structure sqlite cache.")
    p.add_argument("--build-cache-only", action="store_true",
                   help="Build/verify the structure cache, then exit without generating candidates.")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
