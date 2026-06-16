import torch
from gnn_model import MaterialGNN
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
print(f"Project root: {_PROJECT_ROOT}")
GRAPH_PATH = _PROJECT_ROOT / "research" / "phase_2" / "data" / "processed" / "materials_graphs.pt"
MODEL_PATH = _PROJECT_ROOT / "research" / "phase_2" / "models" / "gnn_formation_energy_model.pt"

graphs = torch.load(GRAPH_PATH, weights_only=False)

graph_index = torch.randint(0, len(graphs), (1,))
g = graphs[graph_index]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MaterialGNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

g = g.to(device)
g.batch = torch.zeros(g.x.size(0), dtype=torch.long, device=device)

with torch.no_grad():
    pred = model(g)

print(f"Graph: {g}")
print("Material UID:", g.material_uid)
print("True formation energy:", g.y.item())
print("Predicted formation energy:", pred.item())
print("Absolute error:", abs(g.y.item() - pred.item()))