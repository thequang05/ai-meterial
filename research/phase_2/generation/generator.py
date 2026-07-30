"""
Latent-space generator for novel crystal structures.

Takes a trained GraphVAE checkpoint and produces new material candidates by:
  1. Encoding a set of prototype materials into the latent space
  2. Interpolating / perturbing in latent space (optional)
  3. Decoding the latent vector back into a crystal graph
  4. Scoring each decoded candidate with the pretrained MaterialGNN

The generator is used by the Phase 2 orchestrator (main.py) which wires
it between the Neo4j retrieval step (Phase 1) and the GNN validation step
(Phase 2 / GNN evaluator).

Example:
    python generator.py --checkpoint models/vae_model.pt --n-samples 20
    python generator.py --checkpoint models/vae_model.pt \
        --uids MP_mp-1173034,MP_mp-1100894 \
        --target-energy -2.5
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pymatgen.core import Element

import sys as _sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_GNN_PATH = _PROJECT_ROOT / "research" / "phase_2" / "gnn_model.py"
if not any(str(_PROJECT_ROOT / "research" / "phase_2") in p for p in _sys.path):
    _sys.path.insert(0, str(_PROJECT_ROOT / "research" / "phase_2"))

from graph_vae import GraphVAE, TRAINING_OBJECTIVE_VERSION
from gnn_model import MaterialGNN
from chemistry_validator import ChemicalValidator
from chemistry_filters import symbols_from_atomic_numbers, validate_decoded_atomic_numbers

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_DIR = _PROJECT_ROOT / "research" / "phase_2" / "models"
# Default to the audited composition-group-held-out evaluator.  Keep the
# random-split checkpoint as a separate baseline artifact.
GNN_MODEL_PATH = MODEL_DIR / "gnn_formation_energy_grouped_v1.pt"
GRAPH_PATH = _PROJECT_ROOT / "research" / "phase_2" / "data" / "processed" / "materials_graphs.pt"
DEFAULT_VAE_PATH = MODEL_DIR / "vae_model.pt"
OUTPUT_DIR = _PROJECT_ROOT / "research" / "phase_2" / "generation" / "output"

DEFAULT_LMSTORE_URL = "http://192.168.1.47:1234"


# ── Chemistry constraints for valid materials ────────────────────────────────────

_SYMBOL_TO_ATOMIC_NUMBER = {
    Element.from_Z(atomic_number).symbol: atomic_number
    for atomic_number in range(1, 95)
}


@lru_cache(maxsize=None)
def atomic_symbol(atomic_number: int) -> str:
    """Return a valid element symbol without mutating decoded species."""
    return Element.from_Z(int(atomic_number)).symbol


# ── Graph construction from decoded tensors ──────────────────────────────────

def build_pyg_data(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    atom_types: torch.Tensor,
    uid: str = "generated",
    formation_energy: Optional[float] = None,
) -> torch.Tensor:
    """Assemble a PyG Data object from decoded tensors."""
    from torch_geometric.data import Data
    data = Data(
        x=atom_types.view(-1, 1).long(),
        edge_index=edge_index.long(),
        edge_attr=edge_attr.view(-1, 1).float(),
        material_uid=uid,
    )
    if formation_energy is not None:
        data.y = torch.tensor([formation_energy])
    return data


# ── Latent space interpolation ────────────────────────────────────────────────

def interpolate_latent(
    z_a: torch.Tensor, z_b: torch.Tensor, alpha: float = 0.5
) -> torch.Tensor:
    """
    Linear interpolation between two latent vectors.

    alpha=0.0 → z_a
    alpha=0.5 → midpoint
    alpha=1.0 → z_b
    """
    return (1 - alpha) * z_a + alpha * z_b


def random_perturb_latent(z: torch.Tensor, scale: float = 0.3) -> torch.Tensor:
    """Add Gaussian noise to a latent vector."""
    return z + scale * torch.randn_like(z)


# ── Candidate decoding ────────────────────────────────────────────────────────

def decode_and_validate(
    model: GraphVAE,
    gnn: MaterialGNN,
    z: torch.Tensor,
    device: torch.device,
    prototype_graphs: list,
    edge_threshold: float = 0.5,
    top_k_gnn: int = 5,
    node_temperature: float = 1.0,
    node_mask_rate: float = 0.30,
    required_elements: Optional[list[str]] = None,
    allowed_elements: Optional[list[str]] = None,
    chemical_validator: Optional[ChemicalValidator] = None,
    filter_summary: Optional[dict] = None,
) -> list[dict]:
    """
    Decode latent vectors into candidate materials and score with the GNN.

    Key fix vs. old version:
      - OLD: decode FULL topology from scratch (broken VAE → random graphs)
      - NEW: Use prototype topology + chemistry-aware atom substitutions

    Strategy:
      1. Take prototype structure (proven crystal from Materials Project)
      2. Apply VAE-based atom type perturbation (element substitution)
      3. Enforce chemistry constraints (no radioactive elements, realistic stoichiometry)
      4. Score with pretrained GNN for formation energy

    Args:
        model:           trained GraphVAE (used for latent encoding)
        gnn:             pretrained MaterialGNN (formation-energy predictor)
        z:               [N_samples, latent_dim] latent vectors
        device:          torch device
        prototype_graphs: reference graphs for topology + atom context
        edge_threshold:  Bernoulli threshold (for compatibility, not used in new approach)
        top_k_gnn:       keep only top-k by GNN formation energy
        node_temperature: substitution temperature (higher = more diverse substitutions)
        node_mask_rate: fraction of prototype sites proposed for substitution
    """
    model.eval()
    gnn.eval()

    candidates = []
    rejected = Counter()
    rejected_candidates = 0
    seen_candidate_keys: set[tuple[str, int]] = set()
    required_atomic_numbers = {
        _SYMBOL_TO_ATOMIC_NUMBER[symbol]
        for symbol in (required_elements or [])
        if symbol in _SYMBOL_TO_ATOMIC_NUMBER
    }
    unknown_required = set(required_elements or []) - set(_SYMBOL_TO_ATOMIC_NUMBER)
    if unknown_required:
        raise ValueError(f"Unsupported required element symbols: {sorted(unknown_required)}")

    # Need prototype graphs for realistic topology
    if not prototype_graphs:
        print("[WARN] No prototype graphs provided, using default stable prototypes")
        prototype_graphs = _load_default_prototypes()

    with torch.no_grad():
        total = z.size(0)
        progress_every = max(1, total // 10)
        started = time.perf_counter()
        for i, z_i in enumerate(z):
            z_i = z_i.reshape(1, -1).to(device)  # [1, latent_dim]

            # Get prototype structure (real crystal from Materials Project)
            ref = prototype_graphs[i % len(prototype_graphs)]
            ref = ref.to(device)

            num_nodes = ref.x.size(0)

            # ── Use prototype topology directly ────────────────────────────────
            # Prototype edges are real crystal bonds - guaranteed valid topology
            edge_index = ref.edge_index.clone()
            edge_attr = ref.edge_attr.clone()

            # ── Decode atom types with VAE + temperature sampling ────────────
            # Hide only the sites proposed for substitution. Training uses the
            # same denoising objective, so their true labels are not visible.
            original_atom_types = ref.x.view(-1).long().clamp(1, 94)
            node_mask = torch.rand(num_nodes, device=device) < node_mask_rate
            # Retrieval chemistry is a hard condition, not just a prototype hint.
            # Keep at least the original required-element sites unchanged.
            if required_atomic_numbers:
                required_site_mask = torch.zeros_like(node_mask)
                for atomic_number in required_atomic_numbers:
                    required_site_mask |= original_atom_types == atomic_number
                node_mask &= ~required_site_mask
            if not node_mask.any():
                mutable = torch.where(~torch.isin(
                    original_atom_types,
                    torch.tensor(sorted(required_atomic_numbers), device=device),
                ))[0] if required_atomic_numbers else torch.arange(num_nodes, device=device)
                if mutable.numel() > 0:
                    node_mask[mutable[torch.randint(mutable.numel(), (1,), device=device)]] = True
            encoder_atom_types = original_atom_types.clone()
            encoder_atom_types[node_mask] = 0
            node_embs, _, _ = model.encode_node_embeddings(
                ref, atomic_nums_override=encoder_atom_types
            )

            # Get node logits and apply temperature-based sampling
            node_logits = model.decode_nodes(z_i, node_embs)
            node_logits[:, 0] = -torch.inf

            if node_temperature > 0:
                # Gumbel-softmax sampling for exploration
                sampled_atom_types = torch.argmax(
                    F.gumbel_softmax(node_logits, tau=node_temperature, hard=False),
                    dim=-1,
                )
            else:
                # Deterministic: use most likely atom type
                sampled_atom_types = torch.argmax(node_logits, dim=-1)

            atom_types = original_atom_types.clone()
            atom_types[node_mask] = sampled_atom_types[node_mask].clamp(1, 94)

            # Reject invalid decoded chemistry; never repair it by replacing
            # atoms with O or another convenient species.
            atom_filter = validate_decoded_atomic_numbers(
                atom_types.cpu().tolist(),
                original_atom_types.cpu().tolist(),
                required_elements=required_elements or [],
                allowed_elements=allowed_elements,
            )
            if not atom_filter.passed:
                rejected_candidates += 1
                for reason in atom_filter.reasons:
                    rejected[reason] += 1
                continue

            # Persist the sitewise change, rather than only the reduced
            # formula.  This is the minimum provenance needed to reconstruct
            # the candidate on the prototype's lattice in Task 8.
            substitutions = [
                {
                    "site_index": site_index,
                    "from_Z": int(from_z),
                    "to_Z": int(to_z),
                    "from_symbol": atomic_symbol(int(from_z)),
                    "to_symbol": atomic_symbol(int(to_z)),
                }
                for site_index, (from_z, to_z) in enumerate(
                    zip(original_atom_types.cpu().tolist(), atom_types.cpu().tolist())
                )
                if int(from_z) != int(to_z)
            ]

            syms = symbols_from_atomic_numbers(atom_types.cpu().tolist())
            counts = Counter(syms)
            formula = "".join(f"{symbol}{count if count > 1 else ''}" for symbol, count in counts.most_common())

            # Chemical screening must occur before the learned energy model:
            # an energy prediction for a known or charge-unbalanced formula is
            # not an inverse-design candidate.
            chemistry = (
                chemical_validator.validate(
                    formula,
                    required_elements or [],
                    allowed_elements=allowed_elements,
                )
                if chemical_validator is not None
                else None
            )
            if chemistry is not None and not chemistry.accepted:
                rejected_candidates += 1
                rejected[chemistry.reason or "chemical_validation_failed"] += 1
                continue

            # Fast deterministic dedup before calling the learned energy
            # model.  CIF-level StructureMatcher dedup is applied after Task 8
            # when coordinates are available.
            reduced_formula = chemistry.reduced_formula if chemistry is not None else formula
            dedup_key = (reduced_formula, int(atom_types.size(0)))
            if dedup_key in seen_candidate_keys:
                rejected_candidates += 1
                rejected["duplicate_reduced_formula_and_site_count"] += 1
                continue
            seen_candidate_keys.add(dedup_key)

            # Build PyG data for GNN scoring only after chemistry passes.
            cand_data = build_pyg_data(
                edge_index, edge_attr, atom_types,
                uid=f"gen_{uuid.uuid4().hex[:8]}",
            )
            cand_data.batch = torch.zeros(
                cand_data.x.size(0), dtype=torch.long, device=device
            )
            cand_data = cand_data.to(device)

            # GNN scoring
            gnn_pred = gnn(cand_data).item()

            candidates.append({
                "candidate_id": f"cand_{uuid.uuid4().hex[:8]}",
                "formula": reduced_formula,
                "num_atoms": int(atom_types.size(0)),
                # edge_index is bidirectional, divide by 2 for undirected count
                "num_edges": int(edge_index.size(1) // 2),
                "gnn_formation_energy": round(gnn_pred, 4),
                "latent_alpha": round(float(i / max(len(z) - 1, 1)), 3),
                "prototype_uid": getattr(ref, "material_uid", "unknown"),
                "oxidation_states": chemistry.oxidation_states if chemistry is not None else None,
                "substitutions": substitutions,
            })
            if (i + 1) % progress_every == 0 or i + 1 == total:
                elapsed = time.perf_counter() - started
                print(f"    decoded {i + 1}/{total} ({elapsed:.1f}s)")

    if chemical_validator is not None:
        print(f"  Chemistry screen: accepted {len(candidates)}/{total}; "
              f"rejected {rejected_candidates} {dict(rejected)}")
    if filter_summary is not None:
        filter_summary.clear()
        filter_summary.update({
            "generated": total,
            "passed": len(candidates),
            "rejected": rejected_candidates,
            "rejection_counts": dict(rejected),
            "dedup_key": "reduced_formula+num_sites",
        })

    # Rank by formation energy (most negative = most stable)
    candidates.sort(key=lambda c: c["gnn_formation_energy"])
    # The model can emit the same composition from multiple prototypes.  Keep
    # only the best-scoring representative for a composition-level shortlist.
    unique_candidates = []
    seen_formulas = set()
    for candidate in candidates:
        if candidate["formula"] in seen_formulas:
            continue
        seen_formulas.add(candidate["formula"])
        unique_candidates.append(candidate)
    return unique_candidates[:top_k_gnn]


def _load_default_prototypes() -> list:
    """Load stable prototype materials from dataset."""
    from data_loader import MaterialsGraphDataset

    # Load a subset of low-energy materials as prototypes
    dataset = MaterialsGraphDataset(
        energy_max=-0.5,  # Only stable materials
        max_nodes=32,
        max_edges=256,
    )
    # Return first 20 stable prototypes
    return dataset.graphs[:20]


# ── Main generation ────────────────────────────────────────────────────────────

def load_vae(path: Path, args: argparse.Namespace, device: torch.device) -> GraphVAE:
    state = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(state, dict) or state.get("objective_version") != TRAINING_OBJECTIVE_VERSION:
        raise ValueError(
            f"{path} uses an incompatible GraphVAE objective. "
            f"Expected objective version {TRAINING_OBJECTIVE_VERSION}; retrain from scratch."
        )
    if state.get("latent_gate_passed") is not True:
        raise ValueError(
            f"{path} did not pass the latent-usage gate and cannot be used for generation."
        )
    trained = state.get("args", {})

    def setting(name: str, fallback):
        return trained.get(name, getattr(args, name, fallback))

    model = GraphVAE(
        atom_emb_dim=setting("atom_emb_dim", 96),
        hidden_dim=setting("hidden_dim", 192),
        latent_dim=setting("latent_dim", 64),
        num_gnn_layers=setting("num_layers", 4),
        kl_weight=setting("kl_weight", 0.2),
        node_weight=setting("node_weight", 5.0),
        num_atom_types=setting("num_atom_types", 94),
        edge_pos_weight=setting("edge_pos_weight", 5.0),
        focal_gamma=2.0,
        node_mask_rate=setting("node_mask_rate", 0.30),
        edge_mask_rate=setting("edge_mask_rate", 0.20),
        negative_edge_ratio=setting("negative_edge_ratio", 1.0),
        context_dropout=setting("context_dropout", 0.40),
        kl_free_bits=setting("kl_free_bits", 0.02),
        energy_weight=setting("energy_weight", 1.0),
    )
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device)
    model.eval()
    return model


def load_gnn(device: torch.device, checkpoint_path: Path = GNN_MODEL_PATH) -> MaterialGNN:
    gnn = MaterialGNN()
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        gnn.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    else:
        raise FileNotFoundError(f"GNN checkpoint not found: {checkpoint_path}")
    gnn.to(device)
    gnn.eval()
    return gnn


def generate(
    args: argparse.Namespace,
    vae_model: GraphVAE,
    gnn: MaterialGNN,
    device: torch.device,
) -> list[dict]:
    """
    Core generation pipeline.

    1. If prototype uids are provided, encode them to get latent means.
    2. If a target_energy is given, perform latent optimization or interpolation.
    3. Sample latent vectors and decode them.
    4. Score with GNN and return ranked candidates.
    """
    prototype_graphs: list = []
    latent_means: list = []

    # ── Encode prototype materials (from graph dataset) ───────────────────
    if args.uids:
        graphs_all = torch.load(GRAPH_PATH, weights_only=False)
        uid_to_graph = {}
        for g in graphs_all:
            if hasattr(g, "material_uid"):
                uid_to_graph[g.material_uid] = g

        with torch.no_grad():
            for uid in args.uids:
                if uid in uid_to_graph:
                    g = uid_to_graph[uid].to(device)
                    mu, _ = vae_model.encode(g)
                    latent_means.append(mu.cpu())
                    prototype_graphs.append(uid_to_graph[uid])
                    print(f"  Encoded prototype: {uid}")
                else:
                    print(f"  Prototype {uid} not found in graph dataset — skipping.")

    # ── Build latent vectors ────────────────────────────────────────────────
    if args.interpolate:
        if len(latent_means) >= 2:
            z_list = []
            for i in range(len(latent_means) - 1):
                za = latent_means[i].to(device)
                zb = latent_means[i + 1].to(device)
                for alpha in np.linspace(0, 1, args.samples_per_parent):
                    z_list.append(interpolate_latent(za, zb, float(alpha)))
            z = torch.stack(z_list) if z_list else torch.randn(args.n_samples, args.latent_dim, device=device)
        elif len(latent_means) == 1:
            za = latent_means[0].to(device)
            zb = random_perturb_latent(za, scale=args.perturb_scale)
            z_list = []
            for alpha in np.linspace(0, 1, args.samples_per_parent):
                z_list.append(interpolate_latent(za, zb, float(alpha)))
            z = torch.stack(z_list)
        else:
            z = torch.randn(args.n_samples, args.latent_dim, device=device)
    else:
        if latent_means:
            # Start from prototype means, add perturbation.
            z = torch.cat([lm.to(device) for lm in latent_means], dim=0)
            n_extra = max(0, args.n_samples - z.size(0))
            extra = random_perturb_latent(
                z.mean(dim=0, keepdim=True).expand(n_extra, -1),
                scale=args.perturb_scale,
            )
            z = torch.cat([z, extra], dim=0) if n_extra > 0 else z
        else:
            z = torch.randn(args.n_samples, args.latent_dim, device=device)

    # Truncate to requested number of samples.
    z = z[:args.n_samples]

    # ── Decode + validate with GNN ──────────────────────────────────────────
    node_temp = getattr(args, "node_temperature", 0.7)
    print(f"\nDecoding {z.size(0)} latent vectors (node_temperature={node_temp}) ...")
    candidates = decode_and_validate(
        model=vae_model,
        gnn=gnn,
        z=z,
        device=device,
        prototype_graphs=prototype_graphs,
        edge_threshold=args.edge_threshold,
        top_k_gnn=args.top_k,
        node_temperature=node_temp,
        node_mask_rate=getattr(args, "node_mask_rate", 0.30),
    )

    return candidates


def save_candidates(candidates: list[dict], output_dir: Path) -> Path:
    """Write candidates to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "phase2_candidates.csv"
    df = pd.DataFrame(candidates)
    df.to_csv(manifest_path, index=False)
    print(f"\nSaved {len(candidates)} candidates to {manifest_path}")
    return manifest_path


def run(args: argparse.Namespace) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Generator] Device: {device}")

    # Load models.
    vae_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_VAE_PATH
    print(f"Loading VAE from: {vae_path}")
    vae_model = load_vae(vae_path, args, device)

    print(f"Loading GNN from: {GNN_MODEL_PATH}")
    gnn = load_gnn(device, GNN_MODEL_PATH)

    candidates = generate(args, vae_model, gnn, device)
    save_candidates(candidates, Path(args.output))

    print("\n=== Top Candidates ===")
    for c in candidates:
        print(f"  {c['candidate_id']}: {c['formula']}  "
              f"(E_form={c['gnn_formation_energy']:.3f} eV/atom)  "
              f"atoms={c['num_atoms']} edges={c['num_edges']}")

    return candidates


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Graph VAE generator for novel materials.")
    p.add_argument("--checkpoint", default=str(DEFAULT_VAE_PATH),
                   help="Path to trained VAE checkpoint.")
    p.add_argument("--n-samples", type=int, default=20,
                   help="Total number of latent samples to decode.")
    p.add_argument("--samples-per-parent", type=int, default=5,
                   help="Number of interpolation steps per parent pair.")
    p.add_argument("--interpolate", action="store_true",
                   help="Interpolate between prototype latent means.")
    p.add_argument("--perturb-scale", type=float, default=0.5,
                   help="Gaussian noise scale for latent perturbation.")
    p.add_argument("--uids", type=lambda s: [u.strip() for u in s.split(",") if u.strip()],
                   help="Comma-separated prototype uids for conditioning.")
    p.add_argument("--target-energy", type=float, default=None,
                   help="Desired formation energy (eV/atom) — used for latent selection.")
    p.add_argument("--edge-threshold", type=float, default=0.5,
                   help="Bernoulli threshold for edge existence during decoding.")
    p.add_argument("--top-k", type=int, default=10,
                   help="Return only top-k candidates by GNN score.")
    p.add_argument("--output", default=str(OUTPUT_DIR),
                   help="Output directory for candidates.csv.")
    # Architecture args (must match the trained checkpoint).
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--atom-emb-dim", type=int, default=96)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--kl-weight", type=float, default=0.2)
    p.add_argument("--node-weight", type=float, default=5.0)
    p.add_argument("--node-temperature", type=float, default=0.7,
                   help="Gumbel-Softmax temperature for atom-type sampling during generation. "
                        "0.0=deterministic argmax (safe), 0.5=moderate exploration, "
                        "1.0=maximum diversity. Default 0.7.")
    p.add_argument("--edge-pos-weight", type=float, default=5.0)
    p.add_argument("--node-mask-rate", type=float, default=0.30,
                   help="Fraction of prototype atom sites proposed for substitution.")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
