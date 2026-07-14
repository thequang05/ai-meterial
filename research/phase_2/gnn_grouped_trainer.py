"""Train the formation-energy GNN on the fixed composition-grouped split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from gnn_model import MaterialGNN


ROOT = Path(__file__).resolve().parent.parent.parent
GRAPH_PATH = ROOT / "research" / "phase_2" / "data" / "processed" / "materials_graphs.pt"
SPLIT_PATH = ROOT / "research" / "phase_2" / "data" / "splits" / "materials_broad_v1_seed42.json"
MODEL_DIR = ROOT / "research" / "phase_2" / "models"


@torch.no_grad()
def evaluate(model: MaterialGNN, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    absolute_error = 0.0
    count = 0
    for batch in loader:
        batch = batch.to(device)
        prediction = model(batch)
        target = batch.y.view(-1).float()
        absolute_error += float((prediction - target).abs().sum())
        count += int(target.numel())
    return absolute_error / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, default=SPLIT_PATH)
    parser.add_argument("--checkpoint-name", default="gnn_formation_energy_grouped_v1.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if set(split["train"]) & set(split["val"]) or set(split["train"]) & set(split["test"]):
        raise ValueError("Split manifest has UID overlap.")

    graphs = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    by_uid = {str(graph.material_uid): graph for graph in graphs if hasattr(graph, "material_uid")}
    missing = (set(split["train"]) | set(split["val"])) - set(by_uid)
    if missing:
        raise ValueError(f"{len(missing)} split UIDs are absent from graph data.")
    train_graphs = [by_uid[uid] for uid in split["train"]]
    val_graphs = [by_uid[uid] for uid in split["val"]]

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaterialGNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.L1Loss()
    checkpoint_path = MODEL_DIR / args.checkpoint_name
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_mae = float("inf")
    stale_epochs = 0
    history = []
    print(f"[GNN] device={device}; train={len(train_graphs)}, val={len(val_graphs)}, test={len(split['test'])}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_error = 0.0
        count = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            prediction = model(batch)
            target = batch.y.view(-1).float()
            loss = loss_fn(prediction, target)
            loss.backward()
            optimizer.step()
            total_train_error += float((prediction.detach() - target).abs().sum())
            count += int(target.numel())
        train_mae = total_train_error / max(count, 1)
        val_mae = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_mae": train_mae, "val_mae": val_mae}
        history.append(row)
        print(f"Epoch {epoch:03d} | train MAE={train_mae:.4f} | val MAE={val_mae:.4f}")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            stale_epochs = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "best_val_mae": best_val_mae,
                "split_manifest": str(args.split_manifest),
                "args": {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()},
            }, checkpoint_path)
            print(f"  --> saved {checkpoint_path}")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"[GNN] early stop: no validation improvement for {stale_epochs} epochs")
                break

    history_path = checkpoint_path.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[GNN] best val MAE={best_val_mae:.4f}; history={history_path}")


if __name__ == "__main__":
    main()
