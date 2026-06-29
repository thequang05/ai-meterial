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
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import pandas as pd
from mendeleev import element as mendeleev_element

import sys as _sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_GNN_PATH = _PROJECT_ROOT / "research" / "phase_2" / "gnn_model.py"
if not any(str(_PROJECT_ROOT / "research" / "phase_2") in p for p in _sys.path):
    _sys.path.insert(0, str(_PROJECT_ROOT / "research" / "phase_2"))

from graph_vae import GraphVAE
from gnn_model import MaterialGNN

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_DIR = _PROJECT_ROOT / "research" / "phase_2" / "models"
GNN_MODEL_PATH = MODEL_DIR / "gnn_formation_energy_model.pt"
GRAPH_PATH = _PROJECT_ROOT / "research" / "phase_2" / "data" / "processed" / "materials_graphs.pt"
DEFAULT_VAE_PATH = MODEL_DIR / "vae_model.pt"
OUTPUT_DIR = _PROJECT_ROOT / "research" / "phase_2" / "generation" / "output"

DEFAULT_LMSTUDIO_URL = "http://192.168.1.47:1234"


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
) -> list[dict]:
    """
    Decode one or more latent vectors into candidate materials and score them
    with the GNN.

    Args:
        model:          trained GraphVAE
        gnn:            pretrained MaterialGNN (formation-energy predictor)
        z:              [N_samples, latent_dim] latent vectors
        device:         torch device
        prototype_graphs: list of prototype PyG Data objects (for node/edge counts)
        edge_threshold:  Bernoulli threshold for edge existence
        top_k_gnn:      keep only top-k by GNN-predicted formation energy

    Returns:
        List of candidate dicts: {uid, formula, formation_energy, gnn_score, ...}
    """
    model.eval()
    gnn.eval()

    candidates = []

    # Use prototype graph to get a reference for node count / edge structure.
    # For simplicity: use the first prototype as template for number of nodes.
    if prototype_graphs:
        ref = prototype_graphs[0]
        num_nodes = ref.x.size(0)
        num_edges = ref.edge_index.size(1)
    else:
        num_nodes = z.size(0)  # Will be overridden per sample below
        num_edges = 0

    with torch.no_grad():
        for i, z_i in enumerate(z):
            z_i = z_i.reshape(1, -1).to(device)  # [1, latent_dim]

            # Encode the prototype to get per-node embeddings.
            if prototype_graphs:
                ref = prototype_graphs[i % len(prototype_graphs)]
                ref = ref.to(device)
                result = model(ref, fully_decode=False)
                node_embs = result["node_embs"]
            else:
                # Use a default small graph.
                ref = None
                node_embs = model.atom_embedding(
                    torch.randint(1, model.num_atom_types, (num_nodes,), device=device)
                )

            # ── Decode topology ──────────────────────────────────────────────
            # The edge decoder is driven by the (normalized) bond length, so it
            # cannot synthesize a connectivity from scratch: scoring an O(N²)
            # adjacency with a single constant distance is all-or-nothing. When a
            # prototype graph is available we instead score its *real* candidate
            # edges (with their real bond lengths) and keep the ones the decoder
            # confirms — z then drives novelty through the atom-type decoder below.
            if ref is not None and ref.edge_index.numel() > 0:
                cand_ei = ref.edge_index
                cand_ea = ref.edge_attr if ref.edge_attr is not None else \
                    torch.full((cand_ei.size(1), 1), float(model.edge_norm.shift),
                               device=device)
                logits = model.decode_edges(z_i, node_embs, cand_ei, cand_ea)
                keep = torch.sigmoid(logits) > edge_threshold
                if int(keep.sum()) == 0:
                    continue
                edge_index = cand_ei[:, keep]
                edge_attr = cand_ea[keep]
            else:
                # No prototype topology: fall back to full-adjacency scoring, but
                # feed the learned mean bond length so the decoder is in-distribution.
                all_logits = model.decode_full_adjacency(z_i, node_embs, node_embs.size(0))
                adj = (torch.sigmoid(all_logits) > edge_threshold).float()
                idx = (torch.triu(adj, diagonal=1) > 0.5).nonzero(as_tuple=False)
                if idx.numel() == 0:
                    continue
                ud = idx.t()
                edge_index = torch.cat([ud, ud.flip(0)], dim=1)
                edge_attr = torch.full((edge_index.size(1), 1),
                                       float(model.edge_norm.shift), device=device)

            # ── Decode atom types ─────────────────────────────────────────────
            node_logits = model.decode_nodes(z_i, node_embs)
            atom_types = torch.argmax(node_logits, dim=-1)  # [N]
            atom_types = atom_types.clamp(1, 94)

            # Build PyG data for GNN scoring.
            cand_data = build_pyg_data(
                edge_index, edge_attr, atom_types,
                uid=f"gen_{uuid.uuid4().hex[:8]}",
            )
            cand_data.batch = torch.zeros(cand_data.x.size(0), dtype=torch.long, device=device)
            cand_data = cand_data.to(device)

            # ── GNN scoring ────────────────────────────────────────────────────
            gnn_pred = gnn(cand_data).item()

            # Human-readable formula.
            try:
                from pymatgen.core import Composition
                syms = [mendeleev_element(int(t)).symbol for t in atom_types.cpu()]
                from collections import Counter
                cnt = Counter(syms)
                formula = "".join(f"{s}{c if c > 1 else ''}" for s, c in cnt.most_common())
            except Exception:
                formula = "".join(mendeleev_element(int(t)).symbol for t in atom_types.cpu())

            candidates.append({
                "candidate_id": f"cand_{uuid.uuid4().hex[:8]}",
                "formula": formula,
                "num_atoms": atom_types.size(0),
                "num_edges": edge_index.size(1),
                "gnn_formation_energy": round(gnn_pred, 4),
                "latent_alpha": round(float(i / max(len(z) - 1, 1)), 3),
            })

    # Rank by formation energy (most negative = most stable).
    candidates.sort(key=lambda c: c["gnn_formation_energy"])
    return candidates[:top_k_gnn]


# ── Main generation ────────────────────────────────────────────────────────────

def load_vae(path: Path, args: argparse.Namespace, device: torch.device) -> GraphVAE:
    model = GraphVAE(
        atom_emb_dim=args.atom_emb_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_gnn_layers=args.num_layers,
        kl_weight=args.kl_weight,
        node_weight=args.node_weight,
    )
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device)
    model.eval()
    return model


def load_gnn(device: torch.device) -> MaterialGNN:
    gnn = MaterialGNN()
    if GNN_MODEL_PATH.exists():
        gnn.load_state_dict(torch.load(GNN_MODEL_PATH, map_location=device))
    else:
        print(f"[Warning] GNN model not found at {GNN_MODEL_PATH}, using random init.")
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
    print(f"\nDecoding {z.size(0)} latent vectors ...")
    candidates = decode_and_validate(
        model=vae_model,
        gnn=gnn,
        z=z,
        device=device,
        prototype_graphs=prototype_graphs,
        edge_threshold=args.edge_threshold,
        top_k_gnn=args.top_k,
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
    gnn = load_gnn(device)

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
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--atom-emb-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--kl-weight", type=float, default=0.01)
    p.add_argument("--node-weight", type=float, default=1.0)
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
