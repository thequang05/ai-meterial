"""Audit a formation-energy GNN on the fixed broad group-held-out split."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from gnn_model import MaterialGNN


ROOT = Path(__file__).resolve().parent.parent.parent
GRAPH_PATH = ROOT / "research" / "phase_2" / "data" / "processed" / "materials_graphs.pt"
CSV_PATH = ROOT / "research" / "phase_2" / "data" / "processed" / "materials.csv"
MODEL_PATH = ROOT / "research" / "phase_2" / "models" / "gnn_formation_energy_model.pt"
SPLIT_PATH = ROOT / "research" / "phase_2" / "data" / "splits" / "materials_broad_v1_seed42.json"
REPORT_DIR = ROOT / "research" / "phase_2" / "reports"


def metrics(targets: list[float], predictions: list[float]) -> dict[str, float]:
    errors = [prediction - target for target, prediction in zip(targets, predictions)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    mean_target = sum(targets) / len(targets)
    sst = sum((target - mean_target) ** 2 for target in targets)
    r2 = 1.0 - sum(error * error for error in errors) / sst if sst else float("nan")
    return {"mae_ev_per_atom": mae, "rmse_ev_per_atom": rmse, "r2": r2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH)
    parser.add_argument("--split-manifest", type=Path, default=SPLIT_PATH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-test-graphs", type=int, default=None,
                        help="Deterministic smoke-test cap; omit for the full held-out test set.")
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR / "gnn_audit_broad_v1")
    args = parser.parse_args()

    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    test_uids = set(split["test"])
    train_uids = set(split["train"])
    if test_uids & train_uids:
        raise ValueError("Split manifest leaks UIDs between train and test.")

    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        metadata = {row["uid"]: row for row in csv.DictReader(handle)}
    graphs = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    test_graphs = [graph for graph in graphs if str(graph.material_uid) in test_uids]
    if len(test_graphs) != len(test_uids):
        raise ValueError(f"Found {len(test_graphs)}/{len(test_uids)} test graphs.")

    if args.max_test_graphs is not None:
        if args.max_test_graphs <= 0:
            raise ValueError("--max-test-graphs must be positive")
        test_graphs = test_graphs[:args.max_test_graphs]
    print(f"Auditing {len(test_graphs)} graphs on the fixed test split.", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaterialGNN().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()

    rows = []
    with torch.no_grad():
        loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
        total_batches = len(loader)
        progress_every = max(1, total_batches // 10)
        for batch_index, batch in enumerate(loader, 1):
            batch = batch.to(device)
            predictions = model(batch).detach().cpu().tolist()
            targets = batch.y.view(-1).detach().cpu().tolist()
            for uid, target, prediction in zip(batch.material_uid, targets, predictions):
                source = metadata[str(uid)]
                rows.append({
                    "uid": str(uid),
                    "formula": source["formula"],
                    "elements": source["elements_str"],
                    "num_atoms": int(source["num_atoms"]),
                    "target_ev_per_atom": float(target),
                    "prediction_ev_per_atom": float(prediction),
                    "absolute_error_ev_per_atom": abs(float(prediction) - float(target)),
                })
            if batch_index % progress_every == 0 or batch_index == total_batches:
                print(f"  evaluated {batch_index}/{total_batches} batches", flush=True)

    targets = [row["target_ev_per_atom"] for row in rows]
    predictions = [row["prediction_ev_per_atom"] for row in rows]
    by_size: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = "1-16" if row["num_atoms"] <= 16 else "17-64" if row["num_atoms"] <= 64 else "65+"
        by_size[key].append(row)
    summary = {
        "checkpoint": str(args.checkpoint),
        "split_manifest": str(args.split_manifest),
        "test_count": len(rows),
        "overall": metrics(targets, predictions),
        "by_num_atoms": {
            key: metrics(
                [row["target_ev_per_atom"] for row in group],
                [row["prediction_ev_per_atom"] for row in group],
            ) | {"count": len(group)}
            for key, group in by_size.items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved audit to {args.output_dir}")


if __name__ == "__main__":
    main()
