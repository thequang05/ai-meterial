from torch_geometric.nn import CGConv, global_mean_pool
import torch.nn as nn
import torch.nn.functional as F

class MaterialGNN(nn.Module):
    def __init__(self, atom_emb_dim=64, hidden_dim=64, num_layers=3):
        super().__init__()

        # Atomic number embedding.
        # x contains atomic numbers such as 19, 50, etc.
        self.atom_embedding = nn.Embedding(119, atom_emb_dim)

        self.convs = nn.ModuleList([
            CGConv(channels=atom_emb_dim, dim=1)
            for _ in range(num_layers)
        ])

        self.readout = nn.Sequential(
            nn.Linear(atom_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def encode(self, data):
        # data.x shape: [num_atoms, 1]
        atomic_numbers = data.x.view(-1).long()

        x = self.atom_embedding(atomic_numbers)

        for conv in self.convs:
            x = conv(x, data.edge_index, data.edge_attr)
            x = F.relu(x)

        graph_embedding = global_mean_pool(x, data.batch)

        return graph_embedding

    def forward(self, data):
        graph_embedding = self.encode(data)
        prediction = self.readout(graph_embedding)
        return prediction.view(-1)
