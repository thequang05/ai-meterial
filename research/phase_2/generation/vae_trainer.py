"""
Training script for the Graph VAE.

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
import sys
import time
from pathlib import Path
from graph_vae import GraphVAE
from data_loader import MaterialsGraphDataset, make_vae_loaders

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PYGDataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = _PROJECT_ROOT / "research" / "phase_2" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = MODEL_DIR / "vae_model.pt"


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(args: argparse.Namespace) -> GraphVAE:
    model = GraphVAE(
        atom_emb_dim=args.atom_emb_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_gnn_layers=args.num_layers,
        kl_weight=args.kl_weight,
        node_weight=args.node_weight,
        num_atom_types=args.num_atom_types,
    )
    return model


def get_kl_weight(epoch: int, target_kl: float, warmup_epochs: int) -> float:
    """
    Linear KL annealing: ramp from 0 → target_kl over warmup_epochs.

    This prevents the VAE from immediately collapsing the latent space (large KL
    forces σ → 0 early on) before the reconstruction terms have a chance to learn
    useful representations.
    """
    if epoch <= warmup_epochs:
        return target_kl * epoch / warmup_epochs
    return target_kl


def train_one_epoch(
    model: GraphVAE,
    train_loader: PYGDataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    current_kl_weight: float,
    grad_clip: float = 1.0,
    accumulate: int = 4,
) -> dict[str, float]:
    """Train one epoch with gradient accumulation."""
    model.train()
    total_loss = 0.0
    total_edge = 0.0
    total_node = 0.0
    total_kl = 0.0
    n_batches = 0
    optimizer.zero_grad()

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)

        # Skip graphs with NaN/Inf in features (defensive).
        if not torch.isfinite(data.x).all():
            continue
        if not torch.isfinite(data.edge_attr).all():
            continue

        result = model(data, fully_decode=False)

        # Final safety check on loss inputs.
        for key in ["mu", "logvar", "edge_logits", "node_logits"]:
            if not torch.isfinite(result[key]).all():
                print(f"  [WARN] non-finite {key}, skipping batch {batch_idx}")
                continue

        loss, edge_loss, node_loss, kl = model.vae_loss(
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
    }


@torch.no_grad()
def validate(
    model: GraphVAE,
    val_loader: PYGDataLoader,
    device: torch.device,
    current_kl_weight: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_edge = 0.0
    total_node = 0.0
    total_kl = 0.0
    n_batches = 0

    for data in val_loader:
        data = data.to(device)

        if not torch.isfinite(data.x).all() or not torch.isfinite(data.edge_attr).all():
            continue

        result = model(data, fully_decode=False)

        loss, edge_loss, node_loss, kl = model.vae_loss(
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
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "edge": total_edge / n,
        "node": total_node / n,
        "kl": total_kl / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Graph VAE for material generation.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Gradient-accumulation effective batch size.")
    parser.add_argument("--accumulate", type=int, default=4,
                        help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--atom-emb-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--kl-weight", type=float, default=0.01,
                        help="Target KL weight after warmup.")
    parser.add_argument("--kl-warmup", type=int, default=10,
                        help="Number of epochs for KL annealing (0 = no annealing).")
    parser.add_argument("--node-weight", type=float, default=1.0)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--num-atom-types", type=int, default=94)
    parser.add_argument("--max-nodes", type=int, default=64)
    parser.add_argument("--max-edges", type=int, default=256)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--energy-min", type=float, default=None)
    parser.add_argument("--energy-max", type=float, default=None)
    global args
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Graph VAE Trainer] Device: {device}")
    print(f"  lr={args.lr}, kl_weight={args.kl_weight}, kl_warmup={args.kl_warmup} epochs, "
          f"grad_clip={args.grad_clip}, accumulate={args.accumulate}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = MaterialsGraphDataset(
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        energy_min=args.energy_min,
        energy_max=args.energy_max,
    )

    if len(dataset) == 0:
        print("No graphs loaded — check materials_graphs.pt path.")
        sys.exit(1)

    train_loader, val_loader, _ = make_vae_loaders(
        dataset,
        batch_size=args.batch_size,
        split="stratified",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args).to(device)
    print(f"[Graph VAE] Parameters: {count_parameters(model):,}")
    print(f"  latent_dim={args.latent_dim}, node_weight={args.node_weight}, "
          f"max_nodes={args.max_nodes}, max_edges={args.max_edges}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    start_epoch = 1
    best_val_loss = float("inf")

    # ── Resume ─────────────────────────────────────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[Trainer] Resumed from epoch {start_epoch-1}")

    # ── Training loop ──────────────────────────────────────────────────────────
    header = (
        f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | "
        f"{'Edge':>8} | {'Node':>8} | {'KL':>8} | {'KL_w':>8} | "
        f"{'LR':>10} | {'Time':>8}"
    )
    print("\n" + header)
    print("-" * len(header))

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # KL annealing schedule.
        current_kl = get_kl_weight(epoch, args.kl_weight, args.kl_warmup)

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            current_kl_weight=current_kl,
            grad_clip=args.grad_clip, accumulate=args.accumulate,
        )
        val_metrics = validate(
            model, val_loader, device, current_kl_weight=current_kl,
        )
        scheduler.step()
        elapsed = time.time() - t0

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
            f"{val_metrics['kl']:>8.4f} | "
            f"{current_kl:>8.4f} | "
            f"{scheduler.get_last_lr()[0]:>10.2e} | "
            f"{elapsed:>7.1f}s"
        )

        is_best = math.isfinite(val_metrics["loss"]) and val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]

        if epoch % args.save_every == 0 or is_best:
            ckpt_name = f"vae_best.pt" if is_best else f"vae_epoch{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "args": vars(args),
                },
                MODEL_DIR / ckpt_name,
            )
            if is_best:
                print(f"  --> Best model saved (val_loss={best_val_loss:.4f})")

    print(f"\nTraining done. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
