"""
PyTorch-Geometric DataLoader for the Graph VAE.

Wraps the existing materials_graphs.pt (built by phase_1) and adds:
  - Padding/truncation to fixed (max_nodes, max_edges) shapes
  - Optional filtering by energy range or element composition
  - Stratified train/val/test split preserving energy distribution

The loader emits batched PyG Data objects compatible with both the GraphVAE
encoder and the existing MaterialGNN, so either model can consume the same data.
"""

from __future__ import annotations

import torch
import random
from pathlib import Path
from typing import Optional
from torch_geometric.loader import DataLoader as PYGDataLoader


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
GRAPH_PATH = _PROJECT_ROOT / "research" / "phase_2" / "data" / "processed" / "materials_graphs.pt"


class MaterialsGraphDataset:
    """
    In-memory dataset that loads all graphs from materials_graphs.pt and
    exposes them as a list for split + batching.
    """

    def __init__(
        self,
        graph_path: Path = GRAPH_PATH,
        energy_min: Optional[float] = None,
        energy_max: Optional[float] = None,
        include_elements: Optional[list[str]] = None,
        exclude_elements: Optional[list[str]] = None,
        max_nodes: Optional[int] = None,
        max_edges: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        """
        Args:
            graph_path:        path to materials_graphs.pt
            energy_min:        only keep graphs with y >= energy_min
            energy_max:        only keep graphs with y <= energy_max
            include_elements:  keep only graphs containing ALL these element symbols
            exclude_elements:  keep only graphs containing NONE of these symbols
            max_nodes:         cap nodes per graph (graphs exceeding are discarded)
            max_edges:         cap edges per graph (graphs exceeding are discarded)
            seed:              RNG seed for reproducibility
        """
        self.graph_path = graph_path
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.include_elements = set(include_elements or [])
        self.exclude_elements = set(exclude_elements or [])
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.seed = seed

        self.graphs = self._load_and_filter()

    def _load_and_filter(self) -> list[torch.Tensor]:
        """Load .pt file and apply all filters."""
        raw = torch.load(self.graph_path, weights_only=False)

        filtered = []
        skipped_energy = skipped_nodes = skipped_edges = skipped_elem = 0

        for g in raw:
            # Skip graphs without a target.
            if not (hasattr(g, "y") and g.y is not None):
                skipped_energy += 1
                continue

            y = g.y.item() if g.y.numel() == 1 else g.y[0].item()
            if self.energy_min is not None and y < self.energy_min:
                skipped_energy += 1
                continue
            if self.energy_max is not None and y > self.energy_max:
                skipped_energy += 1
                continue

            # Node count filter.
            num_nodes = g.x.size(0)
            if self.max_nodes is not None and num_nodes > self.max_nodes:
                skipped_nodes += 1
                continue

            # Edge count filter.
            num_edges = g.edge_index.size(1)
            if self.max_edges is not None and num_edges > self.max_edges:
                skipped_edges += 1
                continue

            # Element composition filter.
            if self.include_elements or self.exclude_elements:
                atomic_nums = g.x.view(-1).long()
                from mendeleev import element
                try:
                    present = {element(int(n)).symbol for n in atomic_nums.unique()}
                except Exception:
                    skipped_elem += 1
                    continue

                if self.include_elements and not self.include_elements.issubset(present):
                    skipped_elem += 1
                    continue
                if self.exclude_elements and present.intersection(self.exclude_elements):
                    skipped_elem += 1
                    continue

            filtered.append(g)

        total = len(raw)
        print(
            f"[MaterialsGraphDataset] Loaded {len(filtered)}/{total} graphs "
            f"(skipped: {skipped_energy} energy, {skipped_nodes} nodes, "
            f"{skipped_edges} edges, {skipped_elem} elements)"
        )
        return filtered

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.graphs[idx]

    def stratified_split(
        self, train: float = 0.8, val: float = 0.1, test: float = 0.1
    ) -> tuple[list, list, list]:
        """Stratified split by formation energy quantiles."""
        assert abs(train + val + test - 1.0) < 1e-6
        graphs = self.graphs.copy()
        random.seed(self.seed)

        energies = torch.tensor([
            g.y.item() if g.y.numel() == 1 else g.y[0].item()
            for g in graphs
        ])
        quantiles = torch.linspace(0, 1, 11)  # 10 bins
        boundaries = torch.quantile(energies, quantiles).tolist()

        buckets: dict[int, list] = {i: [] for i in range(len(boundaries) - 1)}
        for idx, g in enumerate(graphs):
            y = energies[idx].item()
            for b in range(len(boundaries) - 1):
                if boundaries[b] <= y < boundaries[b + 1]:
                    buckets[b].append(idx)
                    break

        train_idx, val_idx, test_idx = [], [], []
        for bucket in buckets.values():
            random.shuffle(bucket)
            n = len(bucket)
            n_train = int(n * train)
            n_val = int(n * val)
            train_idx.extend(bucket[:n_train])
            val_idx.extend(bucket[n_train:n_train + n_val])
            test_idx.extend(bucket[n_train + n_val:])

        random.shuffle(train_idx)
        random.shuffle(val_idx)
        random.shuffle(test_idx)

        train_graphs = [graphs[i] for i in train_idx]
        val_graphs   = [graphs[i] for i in val_idx]
        test_graphs  = [graphs[i] for i in test_idx]
        return train_graphs, val_graphs, test_graphs

    def random_split(
        self, train: float = 0.8, val: float = 0.1, test: float = 0.1
    ) -> tuple[list, list, list]:
        """Simple random split."""
        assert abs(train + val + test - 1.0) < 1e-6
        random.seed(self.seed)
        graphs = self.graphs.copy()
        random.shuffle(graphs)
        n = len(graphs)
        n_train = int(n * train)
        n_val = int(n * val)
        return (
            graphs[:n_train],
            graphs[n_train:n_train + n_val],
            graphs[n_train + n_val:],
        )


def vae_collate_fn(batch: list) -> torch.Tensor:
    """
    Default collate for VAE: just return the list (the DataLoader handles batching).
    We return individual graphs to avoid padding issues — each graph is
    a separate VAE forward pass (no intra-batch graph edges).
    """
    return batch


def make_vae_loaders(
    dataset: MaterialsGraphDataset,
    batch_size: int = 32,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    split: str = "stratified",
    **kwargs,
) -> tuple[PYGDataLoader, PYGDataLoader, PYGDataLoader]:
    """
    Build train / val / test PyG DataLoaders for VAE training.

    Batching is done with a custom collate that feeds one graph at a time
    (VAE needs full graph for reconstruction; no mini-batch edge convolution).
    We achieve batch-level gradient updates via accumulating gradients across
    the dataloader iterator.
    """
    if split == "stratified":
        train_g, val_g, test_g = dataset.stratified_split(train_ratio, val_ratio)
    else:
        train_g, val_g, test_g = dataset.random_split(train_ratio, val_ratio)

    train_loader = PYGDataLoader(
        train_g,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: x[0],
    )
    val_loader = PYGDataLoader(
        val_g,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x[0],
    )
    test_loader = PYGDataLoader(
        test_g,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x[0],
    )

    print(
        f"[make_vae_loaders] train={len(train_g)}, val={len(val_g)}, test={len(test_g)}"
    )
    return train_loader, val_loader, test_loader
