import torch
import torch.nn as nn
from pathlib import Path
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from gnn_model import MaterialGNN

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
print(f"Project root: {_PROJECT_ROOT}")
GRAPH_PATH = _PROJECT_ROOT / "research" / "phase_2" / "data" / "processed" / "materials_graphs.pt"
MODEL_PATH = _PROJECT_ROOT / "research" / "phase_2" / "models" / "gnn_formation_energy_model.pt"

if not GRAPH_PATH.exists():
    raise FileNotFoundError(f"Graph file not found at {GRAPH_PATH}")

def main():
    graphs = torch.load(GRAPH_PATH, weights_only=False)

    # Remove invalid graphs if any
    graphs = [
        g for g in graphs
        if hasattr(g, "y") and g.y is not None
    ]

    train_graphs, temp_graphs = train_test_split(
        graphs,
        test_size=0.2,
        random_state=42
    )

    val_graphs, test_graphs = train_test_split(
        temp_graphs,
        test_size=0.5,
        random_state=42
    )

    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MaterialGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()  # MAE loss

    best_val_mae = float("inf")

    for epoch in range(1, 51):
        model.train()
        train_loss = 0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            pred = model(batch)
            target = batch.y.view(-1).float()

            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.num_graphs

        train_mae = train_loss / len(train_graphs)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                pred = model(batch)
                target = batch.y.view(-1).float()

                loss = loss_fn(pred, target)
                val_loss += loss.item() * batch.num_graphs

        val_mae = val_loss / len(val_graphs)

        print(
            f"Epoch {epoch:03d} | "
            f"Train MAE: {train_mae:.4f} | "
            f"Val MAE: {val_mae:.4f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), MODEL_PATH)

    print("Best validation MAE:", best_val_mae)
    print("Model saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()