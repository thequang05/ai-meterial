"""
Fine-tuning trainer for the high-temperature proxy GraphVAE experiment.

Usage:
    python vae_trainer.py
    python vae_trainer.py --epochs 100 --latent-dim 64 --kl-weight 0.05
    python vae_trainer.py --resume checkpoints/vae_epoch50.pt

The trained model (vae_model.pt) is consumed by generator.py to produce
novel material candidates by sampling from the latent space.

Numerical stability measures:
  - KL annealing: kl_weight ramps from 0 → target_kl_weight over warmup_epochs
  - Gradient clipping: max norm = 1.0
  - Gradient accumulation: effective batch_size = batch_size * accumulate
  - Loss normalization: edge loss and node loss are already mean-reduced in graph_vae.py
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path
from graph_vae import GraphVAE, TRAINING_OBJECTIVE_VERSION
from proxy_data_loader import GRAPH_PATH, MaterialsGraphDataset, make_vae_loaders

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PYGDataLoader

# This file lives in research/phase_2/generation/, so models/ is one level up.
_PHASE2_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = _PHASE2_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = MODEL_DIR / "vae_model.pt"


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(
    args: argparse.Namespace,
    energy_mean: float = 0.0,
    energy_std: float = 1.0,
    train_graphs: list | None = None,
) -> GraphVAE:
    # Compute class-frequency weights for node prediction.
    # Rare elements (Ca, Fe) need upweighting because CE loss ignores frequency.
    node_class_weights = _compute_node_weights(train_graphs, num_atom_types=args.num_atom_types)

    model = GraphVAE(
        atom_emb_dim=args.atom_emb_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_gnn_layers=args.num_layers,
        kl_weight=args.kl_weight,
        node_weight=args.node_weight,
        num_atom_types=args.num_atom_types,
        edge_pos_weight=getattr(args, "edge_pos_weight", 5.0),
        focal_gamma=2.0,
        node_class_weights=node_class_weights,
        node_mask_rate=args.node_mask_rate,
        edge_mask_rate=args.edge_mask_rate,
        negative_edge_ratio=args.negative_edge_ratio,
        context_dropout=args.context_dropout,
        kl_free_bits=args.kl_free_bits,
        energy_weight=args.energy_weight,
    )
    model.set_energy_stats(energy_mean, energy_std)
    return model


def compute_energy_stats(graphs: list) -> tuple[float, float]:
    """Compute normalization from the training split only."""
    values = torch.tensor([
        float(g.y.view(-1)[0]) for g in graphs
        if hasattr(g, "y") and g.y is not None
    ], dtype=torch.float32)
    if values.numel() < 2:
        raise ValueError("Need at least two training targets for energy normalization.")
    return float(values.mean()), float(values.std(unbiased=False).clamp_min(1e-6))


def _compute_node_weights(graphs: list | None = None, num_atom_types: int = 94) -> torch.Tensor:
    """
    Compute inverse-frequency weights for node (atom-type) prediction.

    In the dataset: O=39%, Ca=0.3%. Without weighting, the node decoder
    collapses to always predicting O. We use inverse-frequency scaling with
    sqrt smoothing to avoid exploding weights for extremely rare elements.
    """
    if graphs is not None:
        counts = torch.zeros(num_atom_types + 1)
        for graph in graphs:
            values = graph.x.view(-1).long().clamp(0, num_atom_types)
            counts += torch.bincount(values, minlength=num_atom_types + 1).float()
        present = counts > 0
        weights = torch.ones(num_atom_types + 1)
        weights[present] = 1.0 / torch.sqrt(counts[present] / counts[present].sum())
        weights[0] = 1.0
        return weights / (weights.sum() / weights.numel())

    # Representative frequencies from the broad dataset inspection.
    # Key: atomic number → frequency fraction.
    freq = {
        1: 0.036,   # H
        2: 0.001,   # He (very rare)
        3: 0.005,   # Li
        4: 0.002,   # Be
        5: 0.003,   # B
        6: 0.158,   # C
        7: 0.042,   # N
        8: 0.394,   # O ← dominant
        9: 0.018,   # F
        11: 0.005,  # Na
        12: 0.015,  # Mg
        13: 0.055,  # Al
        14: 0.012,  # Si
        15: 0.003,  # P
        16: 0.033,  # S
        17: 0.003,  # Cl
        19: 0.018,  # K
        20: 0.001,  # Ca
        21: 0.001,  # Sc
        22: 0.003,  # Ti
        23: 0.005,  # V
        24: 0.005,  # Cr
        25: 0.002,  # Mn
        26: 0.009,  # Fe
        27: 0.005,  # Co
        28: 0.002,  # Ni
        29: 0.002,  # Cu
        30: 0.005,  # Zn
        31: 0.003,  # Ga
        33: 0.002,  # As
        34: 0.018,  # Se
        38: 0.002,  # Sr
        47: 0.002,  # Ag
        48: 0.003,  # Cd
        49: 0.018,  # In
        50: 0.033,  # Sn
        51: 0.003,  # Sb
        52: 0.003,  # Te
        53: 0.001,  # I
        56: 0.002,  # Ba
        57: 0.001,  # La
        59: 0.001,  # Pr
        60: 0.002,  # Nd
        62: 0.001,  # Sm
        63: 0.001,  # Eu
        64: 0.001,  # Gd
        65: 0.001,  # Tb
        66: 0.001,  # Dy
        67: 0.001,  # Ho
        68: 0.001,  # Er
        69: 0.001,  # Tm
        70: 0.001,  # Yb
        71: 0.001,  # Lu
        80: 0.048,  # Hg
        83: 0.015,  # Bi
        88: 0.001,  # Ra
        90: 0.001,  # Th
        92: 0.001,  # U
    }

    weights = torch.ones(num_atom_types + 1)
    for z, frac in freq.items():
        if 0 < z <= num_atom_types:
            # Inverse sqrt-frequency: upweights rare classes without exploding weights.
            # sqrt smooths extreme ratios (O=0.394 vs Ca=0.001 → ratio 394x → 19.8x after sqrt).
            weights[z] = 1.0 / (frac ** 0.5 + 1e-6)

    # L2-normalize weights to keep loss magnitude reasonable.
    weights = weights / (weights.sum() / weights.numel())
    return weights


def get_kl_weight(
    epoch: int,
    target_kl: float,
    warmup_epochs: int,
    cycle_epochs: int = 0,
) -> float:
    """
    Linear or cyclical KL annealing.

    This prevents the VAE from immediately collapsing the latent space (large KL
    forces σ → 0 early on) before the reconstruction terms have a chance to learn
    useful representations.
    """
    if target_kl <= 0:
        return 0.0
    phase_epoch = epoch
    if cycle_epochs > 0:
        phase_epoch = (epoch - 1) % cycle_epochs + 1
    if warmup_epochs <= 0:
        return target_kl
    progress = min(max((phase_epoch - 1) / max(warmup_epochs - 1, 1), 0.0), 1.0)
    return target_kl * progress


def train_one_epoch(
    model: GraphVAE,
    train_loader: PYGDataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    current_kl_weight: float,
    grad_clip: float = 1.0,
    accumulate: int = 4,
) -> dict[str, float]:
    """Train one epoch with gradient accumulation. VAE processes one graph at a time."""
    model.train()
    total_loss = 0.0
    total_edge = 0.0
    total_node = 0.0
    total_kl = 0.0
    total_kl_objective = 0.0
    total_energy = 0.0
    node_correct = node_total = 0
    edge_correct = edge_total = 0
    n_batches = 0
    optimizer.zero_grad()

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)

        # Skip graphs with NaN/Inf in features (defensive).
        if not torch.isfinite(data.x).all():
            continue
        if not torch.isfinite(data.edge_attr).all():
            continue

        result = model(data, fully_decode=False, corrupt=True)

        # Final safety check on loss inputs.
        valid_result = True
        for key in ["mu", "logvar", "edge_logits", "node_logits", "energy_pred"]:
            if not torch.isfinite(result[key]).all():
                print(f"  [WARN] non-finite {key}, skipping batch {batch_idx}")
                valid_result = False
                break
        if not valid_result:
            continue

        loss, edge_loss, node_loss, kl, kl_objective, energy_loss = model.vae_loss(
            result,
            data,
            kl_weight=current_kl_weight,
            edge_bce_weight=args.edge_weight,
        )

        if not torch.isfinite(loss):
            print(f"  [WARN] NaN/Inf loss at batch {batch_idx}, skipping")
            optimizer.zero_grad()
            continue

        (loss / accumulate).backward()

        if (batch_idx + 1) % accumulate == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(grad_norm):
                print(f"  [WARN] exploding gradients at batch {batch_idx}, skipping step")
                optimizer.zero_grad()
                continue
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        total_edge += edge_loss.item()
        total_node += node_loss.item()
        total_kl += kl.item()
        total_kl_objective += kl_objective.item()
        total_energy += energy_loss.item()
        mask = result["node_mask"]
        node_correct += int((result["node_logits"][mask].argmax(-1) == data.x.view(-1)[mask]).sum())
        node_total += int(mask.sum())
        edge_targets = result["edge_targets"]
        if edge_targets.numel():
            edge_pred = (torch.sigmoid(result["edge_logits"]) >= 0.5).float()
            edge_correct += int((edge_pred == edge_targets).sum())
            edge_total += int(edge_targets.numel())
        n_batches += 1

    # Flush remaining gradients.
    if n_batches % accumulate != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "edge": total_edge / n,
        "node": total_node / n,
        "kl": total_kl / n,
        "kl_objective": total_kl_objective / n,
        "energy": total_energy / n,
        "node_acc": node_correct / max(node_total, 1),
        "edge_acc": edge_correct / max(edge_total, 1),
    }


@torch.no_grad()
def validate(
    model: GraphVAE,
    val_loader: PYGDataLoader,
    device: torch.device,
    current_kl_weight: float,
    latent_probe_scale: float = 1.0,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_edge = 0.0
    total_node = 0.0
    total_kl = 0.0
    total_kl_objective = 0.0
    total_energy = 0.0
    total_energy_abs_error = 0.0
    energy_count = 0
    latent_probability_shift = 0.0
    latent_probability_count = 0
    latent_top1_changes = 0
    node_correct = node_total = 0
    edge_correct = edge_total = 0
    n_batches = 0

    for data in val_loader:
        data = data.to(device)

        if not torch.isfinite(data.x).all() or not torch.isfinite(data.edge_attr).all():
            continue

        result = model(data, fully_decode=False, corrupt=True)

        loss, edge_loss, node_loss, kl, kl_objective, energy_loss = model.vae_loss(
            result,
            data,
            kl_weight=current_kl_weight,
            edge_bce_weight=args.edge_weight,
        )

        if not torch.isfinite(loss):
            continue

        total_loss += loss.item()
        total_edge += edge_loss.item()
        total_node += node_loss.item()
        total_kl += kl.item()
        total_kl_objective += kl_objective.item()
        total_energy += energy_loss.item()
        pred_energy = result["energy_pred"] * model.energy_std + model.energy_mean
        target_energy = data.y.view(-1).float()
        total_energy_abs_error += float((pred_energy - target_energy).abs().sum())
        energy_count += int(target_energy.numel())
        mask = result["node_mask"]
        node_correct += int((result["node_logits"][mask].argmax(-1) == data.x.view(-1)[mask]).sum())
        node_total += int(mask.sum())
        probe_z = (
            result["mu"]
            + latent_probe_scale * torch.randn_like(result["mu"])
        )
        base_probs = torch.softmax(result["node_logits"][mask], dim=-1)
        probe_logits = model.decode_nodes(probe_z, result["node_embs"])[mask]
        probe_probs = torch.softmax(probe_logits, dim=-1)
        latent_probability_shift += float((base_probs - probe_probs).abs().sum())
        latent_probability_count += int(base_probs.numel())
        latent_top1_changes += int(
            (base_probs.argmax(-1) != probe_probs.argmax(-1)).sum()
        )
        edge_targets = result["edge_targets"]
        if edge_targets.numel():
            edge_pred = (torch.sigmoid(result["edge_logits"]) >= 0.5).float()
            edge_correct += int((edge_pred == edge_targets).sum())
            edge_total += int(edge_targets.numel())
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "edge": total_edge / n,
        "node": total_node / n,
        "kl": total_kl / n,
        "kl_objective": total_kl_objective / n,
        "energy": total_energy / n,
        "energy_mae_ev": total_energy_abs_error / max(energy_count, 1),
        "node_acc": node_correct / max(node_total, 1),
        "edge_acc": edge_correct / max(edge_total, 1),
        "latent_prob_shift": latent_probability_shift / max(latent_probability_count, 1),
        "latent_top1_change": latent_top1_changes / max(node_total, 1),
    }


def validate_repeated(
    model: GraphVAE,
    val_loader: PYGDataLoader,
    device: torch.device,
    current_kl_weight: float,
    latent_probe_scale: float,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    """Average stochastic validation probes using fixed, repeatable seeds."""
    if repeats <= 0:
        raise ValueError("latent probe repeats must be positive")
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    runs = []
    try:
        for repeat in range(repeats):
            torch.manual_seed(seed + repeat)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + repeat)
            runs.append(validate(
                model,
                val_loader,
                device,
                current_kl_weight=current_kl_weight,
                latent_probe_scale=latent_probe_scale,
            ))
    finally:
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    averaged = {
        key: sum(run[key] for run in runs) / len(runs)
        for key in runs[0]
    }
    averaged["latent_top1_change_std"] = statistics.pstdev(
        run["latent_top1_change"] for run in runs
    )
    return averaged


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Graph VAE for material generation.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Gradient-accumulation effective batch size.")
    parser.add_argument("--accumulate", type=int, default=4,
                        help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--atom-emb-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--kl-weight", type=float, default=0.2,
                        help="Target beta for the free-bits KL objective.")
    parser.add_argument("--kl-warmup", type=int, default=5,
                        help="Number of epochs for KL annealing (0 = no annealing).")
    parser.add_argument("--kl-cycle", type=int, default=10,
                        help="KL cycle length in epochs (0 = one linear warmup).")
    parser.add_argument("--kl-free-bits", type=float, default=0.02,
                        help="Free KL allowance per latent dimension (nats).")
    parser.add_argument("--node-weight", type=float, default=5.0,
                        help="Upweight node loss 5x because class imbalance dominates "
                             "(O=39%%, Ca=0.3%%) and node decoder was collapsing.")
    parser.add_argument("--edge-pos-weight", type=float, default=5.0,
                        help="Focal loss positive weight for edge prediction.")
    parser.add_argument("--node-mask-rate", type=float, default=0.30,
                        help="Fraction of atom labels hidden before encoding.")
    parser.add_argument("--edge-mask-rate", type=float, default=0.20,
                        help="Fraction of unique links held out before encoding.")
    parser.add_argument("--negative-edge-ratio", type=float, default=1.0,
                        help="Negative candidate links per held-out positive link.")
    parser.add_argument("--context-dropout", type=float, default=0.40,
                        help="Dropout on node context before FiLM decoding.")
    parser.add_argument("--energy-weight", type=float, default=1.0,
                        help="Weight of normalized formation-energy auxiliary loss.")
    parser.add_argument("--latent-probe-scale", type=float, default=1.0,
                        help="Noise scale used for validation latent sensitivity.")
    parser.add_argument("--min-raw-kl", type=float, default=0.05,
                        help="Minimum validation raw KL for a usable checkpoint.")
    parser.add_argument("--max-raw-kl", type=float, default=20.0,
                        help="Maximum validation raw KL for a prior-compatible checkpoint.")
    parser.add_argument("--min-latent-top1-change", type=float, default=0.05,
                        help="Minimum masked-node top-1 change under latent probe.")
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--num-atom-types", type=int, default=94)
    parser.add_argument("--max-nodes", type=int, default=64)
    parser.add_argument("--max-edges", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Rank-stratified graph limit for smoke/pilot runs.")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=str(CHECKPOINT_PATH),
                        help="Canonical best-checkpoint output path.")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--energy-min", type=float, default=None)
    parser.add_argument("--energy-max", type=float, default=None)
    parser.add_argument("--domain-profile", choices=["broad", "high_temperature_proxy_v1"], default="broad")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Load objective-v3 model weights only; optimizer/scheduler are fresh.")
    parser.add_argument("--checkpoint-name", type=str, default=None,
                        help="Model filename under research/phase_2/models (alternative to --checkpoint-path).")
    parser.add_argument("--split-manifest", type=str, default=None,
                        help="Versioned UID split JSON. Required for domain fine-tuning.")
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop after this many latent-healthy non-improving epochs (0 disables).")
    parser.add_argument("--latent-probe-repeats", type=int, default=3,
                        help="Fixed-seed validation probes averaged for checkpoint gating.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-gate-mode", choices=["absolute", "preserve_baseline"],
                        default="preserve_baseline")
    parser.add_argument("--min-latent-retention", type=float, default=0.90,
                        help="Minimum fraction of broad zero-shot Z-change retained.")
    parser.add_argument("--min-domain-latent-top1-change", type=float, default=0.02,
                        help="Hard floor for the domain-relative latent gate.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluate broad --init-from on proxy validation, then exit.")
    global args
    args = parser.parse_args()
    checkpoint_path = MODEL_DIR / args.checkpoint_name if args.checkpoint_name else Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_prefix = (
        f"vae_v{TRAINING_OBJECTIVE_VERSION}"
        if checkpoint_path == CHECKPOINT_PATH
        else checkpoint_path.stem
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Graph VAE Trainer] Device: {device}")
    print(f"  lr={args.lr}, kl_weight={args.kl_weight}, kl_warmup={args.kl_warmup} epochs, "
          f"kl_cycle={args.kl_cycle}, free_bits={args.kl_free_bits}")
    print(f"  context_dropout={args.context_dropout}, energy_weight={args.energy_weight}, "
          f"grad_clip={args.grad_clip}, accumulate={args.accumulate}")
    if args.early_stopping_patience:
        print(f"  early_stopping_patience={args.early_stopping_patience}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    profile_graph_path = (
        _PHASE2_DIR / "data" / "processed" / "high_temperature_proxy_v1_graphs.pt"
        if args.domain_profile == "high_temperature_proxy_v1" else None
    )
    if args.domain_profile != "broad" and not args.split_manifest:
        raise ValueError("Domain fine-tuning requires --split-manifest for reproducibility.")
    if args.init_from and args.resume:
        raise ValueError("Use only one of --init-from or --resume.")
    if args.domain_profile != "broad" and not (args.init_from or args.resume):
        raise ValueError("Domain fine-tuning requires --init-from, or --resume for the same run.")
    dataset = MaterialsGraphDataset(
        graph_path=profile_graph_path or GRAPH_PATH,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        energy_min=args.energy_min,
        energy_max=args.energy_max,
        max_samples=args.max_samples,
    )

    if len(dataset) == 0:
        print("No graphs loaded — check materials_graphs.pt path.")
        sys.exit(1)

    train_loader, val_loader, _ = make_vae_loaders(
        dataset,
        batch_size=args.batch_size,
        split="stratified",
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        split_manifest=Path(args.split_manifest) if args.split_manifest else None,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    energy_mean, energy_std = compute_energy_stats(list(train_loader.dataset))
    model = build_model(args, energy_mean=energy_mean, energy_std=energy_std,
                        train_graphs=list(train_loader.dataset)).to(device)
    print(f"[Graph VAE] Parameters: {count_parameters(model):,}")
    print(f"  latent_dim={args.latent_dim}, node_weight={args.node_weight}, "
          f"max_nodes={args.max_nodes}, max_edges={args.max_edges}")
    print(f"  train energy normalization: mean={energy_mean:.4f}, std={energy_std:.4f} eV/atom")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    start_epoch = 1
    best_val_score = float("inf")
    epochs_without_improvement = 0
    baseline_proxy_metrics = None
    effective_min_latent_change = args.min_latent_top1_change

    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        if ckpt.get("objective_version") != TRAINING_OBJECTIVE_VERSION or ckpt.get("latent_gate_passed") is not True:
            raise ValueError("--init-from requires an objective-v3 checkpoint that passed the latent gate.")
        proxy_node_weights = model.node_class_weights.detach().clone()
        model.load_state_dict(ckpt["model"])
        baseline_proxy_metrics = validate_repeated(
            model,
            val_loader,
            device,
            current_kl_weight=args.kl_weight,
            latent_probe_scale=args.latent_probe_scale,
            repeats=args.latent_probe_repeats,
            seed=args.seed,
        )
        baseline_change = baseline_proxy_metrics["latent_top1_change"]
        if args.latent_gate_mode == "preserve_baseline":
            effective_min_latent_change = max(
                args.min_domain_latent_top1_change,
                min(args.min_latent_top1_change,
                    baseline_change * args.min_latent_retention),
            )
        print(
            "[Proxy baseline] broad zero-shot: "
            f"KL={baseline_proxy_metrics['kl']:.4f}, "
            f"Z-change={baseline_change:.4f}±"
            f"{baseline_proxy_metrics['latent_top1_change_std']:.4f}, "
            f"N-Acc={baseline_proxy_metrics['node_acc']:.3f}, "
            f"E-MAE={baseline_proxy_metrics['energy_mae_ev']:.3f} eV/atom"
        )
        print(
            f"[Proxy gate] mode={args.latent_gate_mode}, "
            f"effective min Z-change={effective_min_latent_change:.4f} "
            f"(absolute reference={args.min_latent_top1_change:.4f})"
        )
        # Evaluate zero-shot with the broad checkpoint's own normalization;
        # switch to proxy-train normalization only for the fine-tuning run.
        model.set_energy_stats(energy_mean, energy_std)
        model.node_class_weights.copy_(proxy_node_weights)
        print(f"[Trainer] Initialized model weights from {args.init_from}; optimizer/scheduler are fresh.")
        if args.eval_only:
            print("[Trainer] Evaluation-only complete; no weights were updated or saved.")
            return

    # ── Resume ─────────────────────────────────────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if ckpt.get("objective_version") != TRAINING_OBJECTIVE_VERSION:
            raise ValueError(
                "Checkpoint uses an incompatible GraphVAE objective and cannot be resumed. "
                "Start a fresh training run."
            )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_val_score = ckpt.get("best_val_score", float("inf"))
        baseline_proxy_metrics = ckpt.get("baseline_proxy_metrics")
        effective_min_latent_change = ckpt.get(
            "effective_min_latent_change", args.min_latent_top1_change
        )
        print(f"[Trainer] Resumed from epoch {start_epoch-1}")

    # ── Training loop ──────────────────────────────────────────────────────────
    header = (
        f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | "
        f"{'Edge':>8} | {'Node':>8} | {'N Acc':>7} | {'E Acc':>7} | "
        f"{'E MAE':>7} | {'KL':>8} | {'Z Chg':>7} | {'Select':>8} | {'KL_w':>8} | "
        f"{'LR':>10} | {'Time':>8}"
    )
    print("\n" + header)
    print("-" * len(header))

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # KL annealing schedule.
        current_kl = get_kl_weight(
            epoch,
            args.kl_weight,
            args.kl_warmup,
            cycle_epochs=args.kl_cycle,
        )

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            current_kl_weight=current_kl,
            grad_clip=args.grad_clip, accumulate=args.accumulate,
        )
        val_metrics = validate_repeated(
            model, val_loader, device, current_kl_weight=current_kl,
            latent_probe_scale=args.latent_probe_scale,
            repeats=args.latent_probe_repeats,
            seed=args.seed,
        )
        scheduler.step()
        elapsed = time.time() - t0
        val_score = (
            val_metrics["edge"]
            + model.node_weight * val_metrics["node"]
            + model.energy_weight * val_metrics["energy"]
        )
        val_metrics["selection_score"] = val_score

        # Check for NaN in val loss.
        val_loss_str = (
            f"{val_metrics['loss']:>12.4f}"
            if math.isfinite(val_metrics["loss"])
            else f"{'NAN':>12}"
        )
        train_loss_str = (
            f"{train_metrics['loss']:>12.4f}"
            if math.isfinite(train_metrics["loss"])
            else f"{'NAN':>12}"
        )

        print(
            f"{epoch:>6} | {train_loss_str} | {val_loss_str} | "
            f"{val_metrics['edge']:>8.4f} | "
            f"{val_metrics['node']:>8.4f} | "
            f"{val_metrics['node_acc']:>7.3f} | "
            f"{val_metrics['edge_acc']:>7.3f} | "
            f"{val_metrics['energy_mae_ev']:>7.3f} | "
            f"{val_metrics['kl']:>8.4f} | "
            f"{val_metrics['latent_top1_change']:>7.3f} | "
            f"{val_score:>8.4f} | "
            f"{current_kl:>8.4f} | "
            f"{scheduler.get_last_lr()[0]:>10.2e} | "
            f"{elapsed:>7.1f}s"
        )

        latent_healthy = (
            val_metrics["kl"] >= args.min_raw_kl
            and val_metrics["kl"] <= args.max_raw_kl
            and val_metrics["latent_top1_change"] >= effective_min_latent_change
        )
        if not latent_healthy:
            print(
                "  [WARN] latent gate failed: "
                f"KL={val_metrics['kl']:.4f} "
                f"(range {args.min_raw_kl}..{args.max_raw_kl}), "
                f"Z-change={val_metrics['latent_top1_change']:.3f} "
                f"(effective min {effective_min_latent_change:.3f})"
            )
        is_best = (
            latent_healthy
            and math.isfinite(val_score)
            and val_score < best_val_score
        )
        if is_best:
            best_val_score = val_score
            epochs_without_improvement = 0
        elif latent_healthy and math.isfinite(best_val_score):
            epochs_without_improvement += 1

        if is_best or epoch % args.save_every == 0:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_score": best_val_score,
                "best_val_loss": val_metrics["loss"] if is_best else None,
                "val_metrics": val_metrics,
                "latent_gate_passed": latent_healthy,
                "absolute_latent_gate_passed": (
                    val_metrics["latent_top1_change"] >= args.min_latent_top1_change
                ),
                "latent_gate_mode": args.latent_gate_mode,
                "effective_min_latent_change": effective_min_latent_change,
                "baseline_proxy_metrics": baseline_proxy_metrics,
                "energy_stats": {"mean": energy_mean, "std": energy_std},
                "args": vars(args),
                "objective_version": TRAINING_OBJECTIVE_VERSION,
            }
            # Best model goes to the canonical path consumed by generator.py / main.py.
            if is_best:
                torch.save(ckpt, checkpoint_path)
                print(
                    f"  --> Best model saved to {checkpoint_path} "
                    f"(selection_score={best_val_score:.4f}, total_val={val_metrics['loss']:.4f})"
                )
            # Periodic snapshot for resuming / inspection.
            if epoch % args.save_every == 0:
                snapshot_path = checkpoint_path.parent / f"{snapshot_prefix}_epoch{epoch}.pt"
                torch.save(ckpt, snapshot_path)

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(f"[Trainer] Early stopping at epoch {epoch}: no selection-score improvement for "
                  f"{epochs_without_improvement} latent-healthy epochs.")
            break

    print(f"\nTraining done. Best selection_score={best_val_score:.4f}")


if __name__ == "__main__":
    main()
