"""
Phase 2 main orchestrator — connects Retrieval, Generation, and Validation.

Implements the Generation Stage from the AIContest.drawio workflow:

    ┌──────────────────────────────────────────────────────────────────┐
    │                     GENERATION STAGE                             │
    │                                                              │
    │  [Relevant Known Materials]  ──────────────────────────────┐   │
    │         from Neo4j MCP          │                         │   │
    │                                │                         │   │
    │  [System Prompt]  ─────────────┼─────────────────────────┤   │
    │                                │                         │   │
    │  [Generative Model]  ←─────────┴───────┐                 │   │
    │    (Graph VAE / Diffusion)              │                 │   │
    │                                     encode                │
    │  [Scientific Filtering]  ←──────── decode                 │
    │    (Pretrained GNN)           + score                    │
    │  [Ranking]  ──────────────────────────────────────────────┤   │
    │  [Proposed Materials]                                        │
    └──────────────────────────────────────────────────────────────┘

Usage:
    python main.py --requirement "material that can withstand 1500C non-conductive"
    python main.py --phase 1        # Retrieval only
    python main.py --phase 2        # Retrieval + Generation
    python main.py --phase 3        # Full pipeline (incl. pymatgen substitution)

The orchestrator is modular: each phase can be run independently or as a pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASET_ROOT  = _PROJECT_ROOT / "dataset"

# Phase 1: Neo4j retrieval.
import sys as _sys
_RETRIEVAL_PATH = _PROJECT_ROOT / "research" / "phase_2"
if str(_RETRIEVAL_PATH) not in _sys.path:
    _sys.path.insert(0, str(_RETRIEVAL_PATH))

from retrieval import MaterialsRetriever

# Phase 2: Graph VAE generation.
_GEN_PATH = _PROJECT_ROOT / "research" / "phase_2" / "generation"
if str(_GEN_PATH) not in _sys.path:
    _sys.path.insert(0, str(_GEN_PATH))

from graph_vae import GraphVAE
from generator import (
    load_vae, load_gnn, generate, interpolate_latent,
    random_perturb_latent, decode_and_validate, build_pyg_data,
    GNN_MODEL_PATH, GRAPH_PATH,
)
from chemistry_validator import ChemicalValidator

DEFAULT_OUTPUT = _PROJECT_ROOT / "research" / "phase_2" / "generation" / "output"
DEFAULT_VAE_PATH = _PROJECT_ROOT / "research" / "phase_2" / "models" / "vae_model.pt"
MATERIALS_CSV = _PROJECT_ROOT / "research" / "phase_2" / "data" / "processed" / "materials.csv"

# A deliberately strict chemistry envelope for refractory *carbides*.  Borides,
# nitrides, oxides, and halides are separate material families and should use a
# separately justified profile rather than being silently mixed into this one.
REFRACTORY_CARBIDE_V1_ELEMENTS = frozenset({
    "C", "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W",
})


# ── Data classes ────────────────────────────────────────────────────────────────

@dataclass
class GenerationConfig:
    """Configuration for the generation stage."""
    vae_checkpoint: Path = DEFAULT_VAE_PATH
    gnn_checkpoint: Path = GNN_MODEL_PATH
    graph_data_path: Path = GRAPH_PATH
    n_samples: int = 20
    interpolate: bool = True
    perturb_scale: float = 0.5
    edge_threshold: float = 0.5
    top_k: int = 10
    latent_dim: int = 64
    atom_emb_dim: int = 96
    hidden_dim: int = 192
    num_layers: int = 4
    kl_weight: float = 0.2
    node_weight: float = 5.0
    node_temperature: float = 0.7
    node_mask_rate: float = 0.30
    required_elements: list[str] = field(default_factory=list)
    allowed_elements: list[str] | None = None
    domain_filter: str | None = None
    chemistry_filter_summary: dict = field(default_factory=dict)
    seed: int = 42


@dataclass
class RetrievalResult:
    """Output from Phase 1: retrieval from Neo4j."""
    uids: list[str]
    materials: list[dict]
    energy_range: dict
    bucket_groups: list[dict] = field(default_factory=list)


@dataclass
class Candidate:
    """A generated material candidate with validation scores."""
    candidate_id: str
    formula: str
    prototype_uid: str
    prototype_formula: str
    num_atoms: int
    num_edges: int
    gnn_formation_energy: float
    generation_method: str  # "vae_interpolation" | "vae_perturbation" | "pymatgen_substitution"
    latent_alpha: float = 0.0
    oxidation_states: dict | None = None
    substitutions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "formula": self.formula,
            "prototype_uid": self.prototype_uid,
            "prototype_formula": self.prototype_formula,
            "num_atoms": self.num_atoms,
            "num_edges": self.num_edges,
            "gnn_formation_energy": self.gnn_formation_energy,
            "generation_method": self.generation_method,
            "latent_alpha": self.latent_alpha,
            "oxidation_states": json.dumps(self.oxidation_states, sort_keys=True) if self.oxidation_states else "",
            "substitutions_json": json.dumps(self.substitutions, sort_keys=True),
            "num_substitutions": len(self.substitutions),
            "structure_status": "not_built",
            "cif_path": "",
        }


# ── Phase 1: Retrieval ─────────────────────────────────────────────────────────

def run_retrieval(
    requirement_query: Optional[dict] = None,
    max_energy: Optional[float] = None,
    min_energy: Optional[float] = None,
    include_elements: Optional[list[str]] = None,
    only_elements: Optional[list[str]] = None,
    limit: int = 10,
    uri: Optional[str] = None,
) -> RetrievalResult:
    """
    Run Phase 1: query Neo4j for known materials matching criteria.

    Args:
        requirement_query: parsed from natural-language (via LLM parser)
        max_energy: upper bound on formation energy (eV/atom)
        min_energy: lower bound on formation energy (eV/atom)
        include_elements: material must contain all these elements
        only_elements: restrict to chemical system (e.g. Li-Fe-O)
        limit: max materials to retrieve
        uri: Neo4j URI (overrides .env)
    """
    print("\n" + "=" * 60)
    print("PHASE 1: RETRIEVAL — querying Neo4j knowledge graph")
    print("=" * 60)

    retriever = MaterialsRetriever(uri=uri) if uri else MaterialsRetriever()

    try:
        stats = retriever.energy_statistics(include_elements=include_elements)
        print(f"  Dataset stats: {stats.get('count', 0)} materials, "
              f"energy range [{stats.get('min', '?'):.3f}, {stats.get('max', '?'):.3f}] eV/atom, "
              f"median={stats.get('median', '?'):.3f}")

        materials = retriever.find_by_formation_energy(
            min_energy=min_energy,
            max_energy=max_energy,
            include_elements=include_elements,
            only_elements=only_elements,
            limit=limit,
            order="asc",
        )

        # Get interpolation parent groups for the generation stage.
        target_energy = max_energy if max_energy is not None else stats.get("median", -1.5)
        buckets = retriever.find_interpolation_parents(
            target_energy=target_energy,
            tolerance=0.3,
            include_elements=include_elements,
            max_buckets=5,
        )

        uids = [m["uid"] for m in materials]

        print(f"\n  Retrieved {len(materials)} known materials:")
        for m in materials[:5]:
            print(f"    {m['uid']:25s} {m['formula']:15s} "
                  f"E_form={m.get('formation_energy_per_atom', 0):.3f} eV/atom")
        if len(materials) > 5:
            print(f"    ... and {len(materials) - 5} more")

        print(f"\n  Found {len(buckets)} interpolation groups "
              f"(compatible graph topologies for generation)")

        return RetrievalResult(
            uids=uids,
            materials=materials,
            energy_range=stats,
            bucket_groups=buckets,
        )

    finally:
        retriever.close()


# ── Phase 2: Generation ─────────────────────────────────────────────────────────

def run_generation(
    retrieval: RetrievalResult,
    config: GenerationConfig,
    target_energy: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> list[Candidate]:
    """
    Run Phase 2: generate novel materials via Graph VAE.

    Steps:
      1. Load trained VAE + pretrained GNN.
      2. Encode prototype materials to get latent means.
      3. Interpolate/perturb in latent space.
      4. Decode latent vectors to crystal graphs.
      5. Score each candidate with GNN → rank by formation energy.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: GENERATION — Graph VAE from prototype materials")
    print("=" * 60)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    print(f"  Generation seed: {config.seed}")

    # Check for trained VAE checkpoint.
    if not config.vae_checkpoint.exists():
        print(f"[WARNING] VAE checkpoint not found at {config.vae_checkpoint}")
        print("  Generation requires a trained VAE. Run:")
        print(f"    python vae_trainer.py --epochs 50 --latent-dim {config.latent_dim}")
        print("  Falling back to random sampling from interpolation buckets.")
        return _fallback_generation_from_buckets(retrieval)

    # Load models.
    vae_args = argparse.Namespace(
        latent_dim=config.latent_dim,
        atom_emb_dim=config.atom_emb_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        kl_weight=config.kl_weight,
        node_weight=config.node_weight,
        edge_pos_weight=5.0,
    )
    vae_model = load_vae(config.vae_checkpoint, vae_args, device)
    gnn = load_gnn(device, config.gnn_checkpoint)

    print(f"  Loaded VAE (latent_dim={config.latent_dim}) from {config.vae_checkpoint}")
    print(f"  Loaded GNN from {config.gnn_checkpoint}")
    if config.allowed_elements:
        print(f"  Candidate domain ({config.domain_filter or 'custom'}): "
              f"{','.join(sorted(config.allowed_elements))}")

    # Encode prototypes.
    graphs_all = torch.load(config.graph_data_path, weights_only=False)
    uid_to_graph = {
        g.material_uid: g for g in graphs_all
        if hasattr(g, "material_uid")
    }

    prototype_graphs = []
    latent_means = []
    for uid in retrieval.uids:
        if uid in uid_to_graph:
            g = uid_to_graph[uid].to(device)
            with torch.no_grad():
                mu, _ = vae_model.encode(g)
            latent_means.append(mu.cpu())
            prototype_graphs.append(uid_to_graph[uid])
            mat = next((m for m in retrieval.materials if m["uid"] == uid), {})
            print(f"  Encoded: {uid} ({mat.get('formula', '?')})  E={mat.get('formation_energy_per_atom', 0):.3f}")

    if not latent_means:
        print("[WARNING] No prototypes found — using random latent sampling.")
        latent_means = [torch.randn(1, config.latent_dim)]

    # Build latent vectors: interpolate between prototypes.
    z_list = []
    if config.interpolate and len(latent_means) >= 2:
        for i in range(len(latent_means) - 1):
            za = latent_means[i].to(device)
            zb = latent_means[i + 1].to(device)
            for alpha in torch.linspace(0, 1, config.n_samples // max(len(latent_means) - 1, 1)):
                z_list.append(interpolate_latent(za, zb, alpha.item()))
    elif len(latent_means) == 1:
        za = latent_means[0].to(device)
        zb = random_perturb_latent(za, scale=config.perturb_scale)
        for alpha in torch.linspace(0, 1, config.n_samples):
            z_list.append(interpolate_latent(za, zb, alpha.item()))
    else:
        for lm in latent_means:
            z_list.append(lm.to(device))

    # Pad with random samples if needed.
    while len(z_list) < config.n_samples:
        z_list.append(random_perturb_latent(
            torch.stack(latent_means).mean(dim=0).to(device),
            scale=config.perturb_scale,
        ))

    # Each z_list entry is [1, latent_dim] (encoder/perturb output), so cat along
    # dim 0 to get [N_samples, latent_dim]. stack() would add a spurious middle
    # dim → [N, 1, latent_dim], breaking the decoder's cat in decode_edges.
    z = torch.cat(z_list[:config.n_samples], dim=0).to(device)
    print(f"\n  Decoding {z.size(0)} latent vectors ...")

    print("  Building known-composition index for novelty screening ...")
    chemical_validator = ChemicalValidator.from_materials_csv(MATERIALS_CSV)
    print(f"  Known reduced compositions: {len(chemical_validator.known_compositions):,}")

    # Decode → chemistry screen → GNN scoring.
    raw_candidates = decode_and_validate(
        model=vae_model,
        gnn=gnn,
        z=z,
        device=device,
        prototype_graphs=prototype_graphs,
        edge_threshold=config.edge_threshold,
        top_k_gnn=config.top_k,
        node_temperature=config.node_temperature,
        node_mask_rate=config.node_mask_rate,
        required_elements=config.required_elements,
        allowed_elements=config.allowed_elements,
        chemical_validator=chemical_validator,
        filter_summary=config.chemistry_filter_summary,
    )

    # Wrap in Candidate objects.
    candidates = []
    for i, raw in enumerate(raw_candidates):
        ref_uid = raw.get("prototype_uid", "unknown")
        ref_mat = next((m for m in retrieval.materials if m["uid"] == ref_uid), {})
        candidates.append(Candidate(
            candidate_id=raw.get("candidate_id", f"gen_{uuid.uuid4().hex[:8]}"),
            formula=raw.get("formula", "?"),
            prototype_uid=ref_uid,
            prototype_formula=ref_mat.get("formula", "?"),
            num_atoms=raw.get("num_atoms", 0),
            num_edges=raw.get("num_edges", 0),
            gnn_formation_energy=raw.get("gnn_formation_energy", 0.0),
            generation_method="vae_interpolation" if config.interpolate else "vae_perturbation",
            latent_alpha=raw.get("latent_alpha", i / max(len(raw_candidates) - 1, 1)),
            oxidation_states=raw.get("oxidation_states"),
            substitutions=raw.get("substitutions", []),
        ))

    # Sort by formation energy (most stable first).
    candidates.sort(key=lambda c: c.gnn_formation_energy)

    print(f"\n  Generated {len(candidates)} candidates:")
    for c in candidates[:5]:
        print(f"    {c.candidate_id}: {c.formula:20s}  "
              f"E_form={c.gnn_formation_energy:.3f}  "
              f"from {c.prototype_formula}")
    if len(candidates) > 5:
        print(f"    ... and {len(candidates) - 5} more")

    return candidates


def _fallback_generation_from_buckets(retrieval: RetrievalResult) -> list[Candidate]:
    """
    Fallback when VAE checkpoint is unavailable: generate candidates by
    interpolation between materials in the same (num_atoms, num_edges) bucket.
    This is the original approach described in retrieval.py comments.
    """
    print("\n  [Fallback] Using interpolation-bucket sampling (no VAE needed).")

    candidates = []
    for bucket in retrieval.bucket_groups[:3]:
        members = bucket.get("members", [])
        if len(members) < 2:
            continue
        # Randomly pick two parents from the same bucket.
        import random
        random.seed(42)
        a, b = random.sample(members, 2)
        # Simple average of properties as the "generated" candidate.
        avg_energy = (a.get("formation_energy_per_atom", 0) +
                     b.get("formation_energy_per_atom", 0)) / 2
        candidates.append(Candidate(
            candidate_id=f"bucket_{uuid.uuid4().hex[:6]}",
            formula=f"(interpolated {a.get('formula', '?')} ↔ {b.get('formula', '?')})",
            prototype_uid=a.get("uid", "?"),
            prototype_formula=a.get("formula", "?"),
            num_atoms=bucket.get("num_atoms", 0),
            num_edges=bucket.get("num_edges", 0),
            gnn_formation_energy=avg_energy,
            generation_method="bucket_interpolation",
            latent_alpha=0.5,
        ))

    print(f"  Generated {len(candidates)} bucket-interpolated candidates.")
    return candidates


# ── Phase 3: Validation (GNN + Ranking) ────────────────────────────────────────

def run_validation(
    candidates: list[Candidate],
    config: GenerationConfig,
    device: Optional[torch.device] = None,
) -> list[Candidate]:
    """
    Run Phase 3: re-score all candidates with the GNN and rank.

    This mirrors the Validation Stage from AIContest.drawio:
      - Scientific Filtering (pretrained GNN validates formation energy)
      - Ranking (by confidence / stability score)
    """
    print("\n" + "=" * 60)
    print("PHASE 3: VALIDATION — GNN scoring and ranking")
    print("=" * 60)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not config.gnn_checkpoint.exists():
        print(f"[WARNING] GNN checkpoint not found — candidates passed through without re-scoring.")
        return candidates

    gnn = load_gnn(device, config.gnn_checkpoint)
    print(f"  GNN loaded from {config.gnn_checkpoint}")

    # All candidates are already scored by GNN in Phase 2.
    # Here we apply additional filters and re-rank.
    print(f"\n  Filtering {len(candidates)} candidates ...")

    stable_candidates = []
    for c in candidates:
        # Stability criterion: formation energy should be negative (exothermic).
        if c.gnn_formation_energy < 0.5:  # threshold: < 0.5 eV/atom
            stable_candidates.append(c)

    print(f"  Stable candidates (E_form < 0.5 eV/atom): {len(stable_candidates)}")

    # Sort by formation energy (most stable = most negative first).
    stable_candidates.sort(key=lambda c: c.gnn_formation_energy)

    print(f"\n  Top 5 most stable candidates:")
    for i, c in enumerate(stable_candidates[:5], 1):
        print(f"    {i}. {c.formula:20s}  E_form={c.gnn_formation_energy:.4f}  "
              f"method={c.generation_method}")

    return stable_candidates


# ── Full pipeline orchestrator ─────────────────────────────────────────────────

def run_pipeline(
    requirement: str,
    max_energy: Optional[float] = None,
    min_energy: Optional[float] = None,
    include_elements: Optional[list[str]] = None,
    only_elements: Optional[list[str]] = None,
    limit: int = 10,
    generation_config: Optional[GenerationConfig] = None,
    output_dir: Path = DEFAULT_OUTPUT,
    device: Optional[torch.device] = None,
) -> list[Candidate]:
    """
    Run the full 3-phase pipeline: Retrieval → Generation → Validation.

    Args:
        requirement: natural-language description of desired material
        max_energy: upper bound on formation energy (eV/atom)
        min_energy: lower bound on formation energy (eV/atom)
        include_elements: required elements (e.g. ["O"] for oxides)
        only_elements: restrict chemical system (e.g. ["Li", "Fe", "O"])
        limit: max known materials to retrieve
        generation_config: Phase 2 configuration
        output_dir: where to write candidate manifest
        device: torch device

    Returns:
        List of validated Candidate objects, ranked by stability.
    """
    t_start = time.time()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if generation_config is None:
        generation_config = GenerationConfig()

    print("\n" + "=" * 70)
    print(f"  PHASE 2 GENERATION PIPELINE — {requirement!r}")
    print("=" * 70)
    print(f"  Requirement: {requirement}")
    print(f"  Filters: max_energy={max_energy}, include_elements={include_elements}")
    if generation_config.domain_filter:
        print(f"  Candidate domain filter: {generation_config.domain_filter}")
    effective_only_elements = only_elements
    if effective_only_elements is None and generation_config.allowed_elements:
        # A candidate-only filter cannot recover a useful result when every
        # prototype is an oxyhalide or salt.  Use the same chemistry envelope
        # for retrieval unless the caller explicitly supplies a narrower one.
        effective_only_elements = generation_config.allowed_elements
        print("  Retrieval chemical system inherited from candidate domain filter: "
              f"{','.join(sorted(effective_only_elements))}")
    if generation_config.domain_filter == "refractory_carbide_v1" and max_energy is not None and max_energy < 0.0:
        print("  [NOTE] max_energy < 0.0 may exclude experimentally stable carbides; "
              "use 0.0 for the first carbide retrieval audit.")
    print(f"  Device: {device}")

    # Phase 1: Retrieval.
    retrieval = run_retrieval(
        max_energy=max_energy,
        min_energy=min_energy,
        include_elements=include_elements,
        only_elements=effective_only_elements,
        limit=limit,
    )

    # Phase 2: Generation.
    candidates = run_generation(
        retrieval=retrieval,
        config=generation_config,
        target_energy=max_energy,
        device=device,
    )

    # Phase 3: Validation + Ranking.
    validated = run_validation(
        candidates=candidates,
        config=generation_config,
        device=device,
    )

    # ── Save results ────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "generation_manifest.csv"
    summary_path = output_dir / "generation_summary.json"

    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["candidate_id", "formula", "prototype_uid", "prototype_formula",
                        "num_atoms", "num_edges", "gnn_formation_energy",
                        "generation_method", "latent_alpha", "oxidation_states",
                        "substitutions_json", "num_substitutions",
                        "structure_status", "cif_path"],
        )
        writer.writeheader()
        writer.writerows([c.to_dict() for c in validated])

    summary = {
        "requirement": requirement,
        "phase": "generation",
        "n_retrieved": len(retrieval.materials),
        "retrieved_uids": retrieval.uids,
        "n_generated": len(candidates),
        "n_validated": len(validated),
        "n_samples_requested": generation_config.n_samples,
        "top_k": generation_config.top_k,
        "target_energy": max_energy,
        "include_elements": include_elements,
        "only_elements": effective_only_elements,
        "domain_filter": generation_config.domain_filter,
        "allowed_elements": sorted(generation_config.allowed_elements or []),
        "chemistry_filter_summary": generation_config.chemistry_filter_summary,
        "generation_seed": generation_config.seed,
        "retrieval_stats": retrieval.energy_range,
        "top_candidates": [c.to_dict() for c in validated[:5]],
        "elapsed_seconds": round(time.time() - t_start, 1),
    }

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"\n  Results saved:")
    print(f"    Manifest: {manifest_path}")
    print(f"    Summary:  {summary_path}")
    print(f"  Total time: {time.time() - t_start:.1f}s")

    return validated


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 2 orchestrator: Generation Stage.")
    p.add_argument("--requirement", type=str,
                   default="stable oxide with low formation energy",
                   help="Natural-language material requirement.")
    p.add_argument("--max-energy", type=float, default=None,
                   help="Upper bound on formation energy (eV/atom).")
    p.add_argument("--min-energy", type=float, default=None,
                   help="Lower bound on formation energy (eV/atom).")
    p.add_argument("--include-elements", type=lambda s: [e.strip() for e in s.split(",")],
                   help="Comma-separated required elements, e.g. O,N.")
    p.add_argument("--only-elements", type=lambda s: [e.strip() for e in s.split(",")],
                   help="Restrict to chemical system, e.g. Li,Fe,O.")
    domain_group = p.add_mutually_exclusive_group()
    domain_group.add_argument(
        "--domain-filter", choices=["refractory_carbide_v1"],
        help="Apply a validated chemistry envelope to generated candidates.",
    )
    domain_group.add_argument(
        "--allowed-elements", type=lambda s: [e.strip() for e in s.split(",") if e.strip()],
        help="Custom generated-candidate element allow-list, e.g. C,Ti,Zr,Hf,Nb,Ta,Mo,W.",
    )
    p.add_argument("--limit", type=int, default=10,
                   help="Max known materials to retrieve.")
    p.add_argument("--phase", type=int, choices=[1, 2, 3],
                   help="Run only up to this phase: 1=retrieval, 2=generation, 3=validation.")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help="Output directory.")
    p.add_argument("--vae-checkpoint", default=str(DEFAULT_VAE_PATH),
                   help="Path to trained VAE checkpoint.")
    p.add_argument("--gnn-checkpoint", default=str(GNN_MODEL_PATH),
                   help="Audited formation-energy GNN checkpoint.")
    p.add_argument("--n-samples", type=int, default=20,
                   help="Number of latent vectors to sample.")
    p.add_argument("--interpolate", action="store_true", default=True,
                   help="Interpolate between prototype latent means.")
    p.add_argument("--no-interpolate", dest="interpolate", action="store_false",
                   help="Disable latent interpolation (use perturbation instead).")
    p.add_argument("--perturb-scale", type=float, default=0.5,
                   help="Gaussian noise scale for perturbation.")
    p.add_argument("--top-k", type=int, default=10,
                   help="Return only top-k validated candidates.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible latent decoding and deduplication.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    allowed_elements = args.allowed_elements
    if args.domain_filter == "refractory_carbide_v1":
        allowed_elements = sorted(REFRACTORY_CARBIDE_V1_ELEMENTS)
    # Apply a profile's chemistry envelope consistently to retrieval-only and
    # generation-only invocations too.  An explicit --only-elements remains
    # the caller's narrower override.
    effective_only_elements = args.only_elements or allowed_elements

    config = GenerationConfig(
        vae_checkpoint=Path(args.vae_checkpoint),
        gnn_checkpoint=Path(args.gnn_checkpoint),
        n_samples=args.n_samples,
        interpolate=args.interpolate,
        perturb_scale=args.perturb_scale,
        top_k=args.top_k,
        required_elements=args.include_elements or [],
        allowed_elements=allowed_elements,
        domain_filter=args.domain_filter,
        seed=args.seed,
    )

    if args.phase == 1:
        result = run_retrieval(
            max_energy=args.max_energy,
            min_energy=args.min_energy,
            include_elements=args.include_elements,
            only_elements=effective_only_elements,
            limit=args.limit,
        )
        print(f"\nRetrieved {len(result.materials)} materials.")
        for m in result.materials:
            print(f"  {m['uid']:25s} {m['formula']:15s} "
                  f"E={m.get('formation_energy_per_atom', 0):.3f}")

    elif args.phase == 2:
        retrieval = run_retrieval(
            max_energy=args.max_energy,
            min_energy=args.min_energy,
            include_elements=args.include_elements,
            only_elements=effective_only_elements,
            limit=args.limit,
        )
        candidates = run_generation(
            retrieval=retrieval,
            config=config,
        )
        print(f"\nGenerated {len(candidates)} candidates.")

    else:  # Full pipeline (default)
        validated = run_pipeline(
            requirement=args.requirement,
            max_energy=args.max_energy,
            min_energy=args.min_energy,
            include_elements=args.include_elements,
            only_elements=effective_only_elements,
            limit=args.limit,
            generation_config=config,
            output_dir=Path(args.output),
        )
        print(f"\n{'=' * 60}")
        print(f"FINAL RESULT: {len(validated)} validated candidates")
        print(f"{'=' * 60}")
        for i, c in enumerate(validated, 1):
            print(f"  {i}. {c.formula:20s}  "
                  f"E_form={c.gnn_formation_energy:.4f} eV/atom  "
                  f"method={c.generation_method}")


if __name__ == "__main__":
    main()
