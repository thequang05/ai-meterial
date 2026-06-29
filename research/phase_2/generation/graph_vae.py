"""
Graph Variational Autoencoder (Graph VAE) for crystal structure generation.

Architecture follows the 2018 Simonovsky et al. design:
  - Encoder: CGConv message-passing → LayerNorm → FC → (mu, logvar)
  - Decoder 1 (edge): MLP(edge_emb ⊕ u_i ⊕ u_j) → Bernoulli per candidate edge
  - Decoder 2 (node):  MLP(node_emb ⊕ z) → categorical per atom type

The latent space Z is the space of "materials fingerprints" from which new
crystal graphs can be decoded.  The GNN (phase_2/gnn_model.py) independently
predicts formation energy from the same node/edge features, so Z also encodes
the thermodynamic landscape — a crucial inductive bias for inverse design.

Training loss:
  L = L_RECON_EDGE + β * L_RECON_NODE + α * L_KL

Numerical stability measures:
  - LayerNorm after each CGConv layer → stable activation scales
  - LayerNorm on concatenated decoder inputs → bounded logits range
  - Edge attr normalization: (x - mean) / std via learnable affine transform
  - KL clipping: logvar clamped to [-10, 10], per-dim KL capped at 1.0
  - Edge logits clipped to [-10, 10] before sigmoid → no log(0) overflow
  - Label smoothing (0.05) for BCE → no saturated gradients
  - Xavier uniform init with gain=1.414 for all Linear layers
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_mean_pool
from typing import Tuple


# ── Initialization ─────────────────────────────────────────────────────────────

def _xavier_init(m: nn.Module) -> None:
    """Xavier uniform for Linear layers, N(0, 0.02) for Embeddings."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)


# ── Edge attribute normalizer ──────────────────────────────────────────────────

class EdgeAttrNorm(nn.Module):
    """
    Learnable per-dimension normalization for edge attributes (bond lengths).
    Normalizes (x - mean) / std using running stats estimated from data,
    plus a small eps to avoid division by zero.
    The data diagnostic shows edge_attr ∈ [0.73, 5.0] Å.
    """

    def __init__(self, dim: int = 1, init_mean: float = 3.83, init_std: float = 0.88) -> None:
        super().__init__()
        self.eps = 1e-4
        # Learnable shift and scale so the model can undo normalization if needed.
        self.shift = nn.Parameter(torch.zeros(dim) + init_mean)
        self.scale = nn.Parameter(torch.ones(dim) * init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to roughly N(0, 1).
        return (x - self.shift) / (self.scale + self.eps)


# ── Graph VAE ──────────────────────────────────────────────────────────────────

class GraphVAE(nn.Module):
    """
    Full VAE: encoder + edge decoder + node decoder.

    Edge decoding is unconditioned (generates graph topology from latent z).
    Node decoding is conditioned on z (generates atom types per node).

    Decoder input dimensions:
      edge_in = 2*atom_emb_dim (i_emb + j_emb) + 1 (ea_norm) + latent_dim (z)
    """

    def __init__(
        self,
        atom_emb_dim: int = 64,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_gnn_layers: int = 3,
        kl_weight: float = 0.01,
        node_weight: float = 1.0,
        num_atom_types: int = 94,
        max_kl: float = 1.0,
        edge_logits_clip: float = 10.0,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.node_weight = node_weight
        self.num_atom_types = num_atom_types
        self.max_kl = max_kl
        self.edge_logits_clip = edge_logits_clip
        self.label_smoothing = label_smoothing
        self.LOGVAR_CLIP = 10.0

        # ── Edge attribute normalization ──────────────────────────────────────
        self.edge_norm = EdgeAttrNorm(dim=1)

        # ── Shared embedding layer ────────────────────────────────────────────
        self.atom_embedding = nn.Embedding(num_atom_types + 1, atom_emb_dim)

        # ── Message-passing encoder ──────────────────────────────────────────
        self.convs = nn.ModuleList([
            CGConv(channels=atom_emb_dim, dim=1)
            for _ in range(num_gnn_layers)
        ])
        # LayerNorm after each CGConv to stabilize activation scales.
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(atom_emb_dim)
            for _ in range(num_gnn_layers)
        ])

        # ── Latent projection (h → μ, σ) ────────────────────────────────────
        self.fc_mu = nn.Linear(atom_emb_dim, latent_dim)
        self.fc_logvar = nn.Linear(atom_emb_dim, latent_dim)

        # ── Edge decoder: MLP(u_i ⊕ u_j ⊕ e_ij ⊕ z) → p(edge) ─────────────
        # 2*atom_emb_dim (i_emb + j_emb) + 1 (ea_norm) + latent_dim (z_expanded)
        edge_in = 2 * atom_emb_dim + 1 + latent_dim
        self.edge_ln = nn.LayerNorm(edge_in)
        self.edge_decoder = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # ── Node decoder: MLP(atom_emb_i ⊕ z) → atom-type logits ────────────
        node_in = atom_emb_dim + latent_dim
        self.node_ln = nn.LayerNorm(node_in)
        self.node_decoder = nn.Sequential(
            nn.Linear(node_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_atom_types + 1),
        )

        # ── Apply weight initialization ────────────────────────────────────────
        self.apply(_xavier_init)

        # Edge decoder last layer: init to near-zero so sigmoid starts near 0.5.
        nn.init.zeros_(self.edge_decoder[-1].weight)
        nn.init.zeros_(self.edge_decoder[-1].bias)

    # ── KL with clipping ──────────────────────────────────────────────────────

    def _kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL divergence D_KL(N(μ,σ) ‖ N(0,I)), clipped to prevent overflow.

        - Clamp logvar to [-LOGVAR_CLIP, LOGVAR_CLIP] before exp().
        - Cap per-dimension KL at max_kl to bound the gradient.
        """
        logvar_clamped = logvar.clamp(-self.LOGVAR_CLIP, self.LOGVAR_CLIP)
        kl_per_dim = 0.5 * (
            logvar_clamped.exp() + mu.pow(2) - 1 - logvar_clamped
        )
        return kl_per_dim.clamp(max=self.max_kl).sum(dim=-1).mean()

    # ── Encoder ───────────────────────────────────────────────────────────────

    def encode(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run message-passing and return μ and log σ² of the variational posterior.

        Args:
            data: PyG Data object with x [N,1] (atomic numbers), edge_index,
                  edge_attr [E,1] (bond lengths), batch [N].

        Returns:
            mu:      [batch_size, latent_dim]
            logvar:  [batch_size, latent_dim]
        """
        atomic_nums = data.x.view(-1).long().clamp(0, self.num_atom_types)
        x = self.atom_embedding(atomic_nums)

        # Normalize edge attributes once per forward pass.
        ea_norm = self.edge_norm(data.edge_attr)

        for conv, ln in zip(self.convs, self.layer_norms):
            x_new = conv(x, data.edge_index, ea_norm)
            x = ln(x_new + x)  # residual-like: x = ln(conv_out + x)

        # Global pooling → one graph embedding per batch element.
        graph_emb = global_mean_pool(x, data.batch)  # [B, atom_emb_dim]

        mu = self.fc_mu(graph_emb)
        logvar = self.fc_logvar(graph_emb)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = μ + σ * ε,  ε ~ N(0, I)."""
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    # ── Edge decoder ──────────────────────────────────────────────────────────

    def decode_edges(
        self,
        z: torch.Tensor,
        node_embs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score every candidate edge (i, j) with the MLP and return logits.

        Args:
            z:          [B, latent_dim]  — same z for all nodes in a graph
            node_embs:  [N, atom_emb_dim] — per-node embeddings from encoder
            edge_index: [2, E]
            edge_attr:  [E, 1] — raw bond lengths (will be normalized)

        Returns:
            edge_logits: [E] — Bernoulli logits for existing edges
        """
        B = z.size(0)
        if B == 1:
            batch_idx = edge_index.new_zeros(edge_index.size(1), dtype=torch.long)
        else:
            batch_idx = torch.repeat_interleave(
                torch.arange(B, device=edge_index.device),
                torch.bincount(edge_index[0].cpu(), minlength=B).to(edge_index.device),
            )

        z_expanded = z[batch_idx]                          # [E, latent_dim]
        ea_norm = self.edge_norm(edge_attr[:, :1])          # [E, 1]
        i_emb = node_embs[edge_index[0]]                   # [E, atom_emb_dim]
        j_emb = node_embs[edge_index[1]]                   # [E, atom_emb_dim]

        # LayerNorm before MLP → bounded input range regardless of activation magnitude.
        edge_in = self.edge_ln(torch.cat([i_emb, j_emb, ea_norm, z_expanded], dim=-1))
        logits = self.edge_decoder(edge_in).squeeze(-1)    # [E]

        # Clip logits to prevent sigmoid saturation → log(0) in BCE.
        return logits.clamp(-self.edge_logits_clip, self.edge_logits_clip)

    def decode_full_adjacency(
        self,
        z: torch.Tensor,
        node_embs: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Score all O(N²) upper-triangular candidate edges.

        Returns:
            all_logits: [N, N] symmetric logits (lower triangle = -inf)
        """
        B = z.size(0)
        device = node_embs.device
        all_logits = torch.full((num_nodes, num_nodes), -1e9, device=device)

        rows, cols = [], []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                rows.append(i)
                cols.append(j)

        if not rows:
            return all_logits

        edge_index = torch.tensor([rows, cols], device=device)
        # Feed the learned mean bond length (not 1.0 Å, which is far off-distribution
        # for EdgeAttrNorm and forces every logit to the clamp floor → zero edges).
        dummy_attr = torch.full(
            (len(rows), 1), float(self.edge_norm.shift), device=device
        )

        logits = self.decode_edges(z, node_embs, edge_index, dummy_attr)
        all_logits[rows, cols] = logits
        all_logits[cols, rows] = logits
        return all_logits

    # ── Node decoder ──────────────────────────────────────────────────────────

    def decode_nodes(
        self, z: torch.Tensor, node_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict atom-type distribution for each node.

        Args:
            z:         [B, latent_dim]
            node_embs: [N, atom_emb_dim]

        Returns:
            node_logits: [N, num_atom_types + 1]
        """
        N = node_embs.size(0)
        B = z.size(0)
        z_expanded = z.expand(N, -1) if B == 1 else z.repeat(N, 1)
        node_in = self.node_ln(torch.cat([node_embs, z_expanded], dim=-1))
        return self.node_decoder(node_in)

    # ── Full forward ──────────────────────────────────────────────────────────

    def forward(
        self, data, fully_decode: bool = True
    ) -> dict[str, torch.Tensor]:
        """Run full VAE forward pass."""
        atomic_nums = data.x.view(-1).long().clamp(0, self.num_atom_types)
        x = self.atom_embedding(atomic_nums)
        ea_norm = self.edge_norm(data.edge_attr)

        for conv, ln in zip(self.convs, self.layer_norms):
            x_new = conv(x, data.edge_index, ea_norm)
            x = ln(x_new + x)

        node_embs = x
        graph_emb = global_mean_pool(node_embs, data.batch)

        mu = self.fc_mu(graph_emb)
        logvar = self.fc_logvar(graph_emb)
        z = self.reparameterize(mu, logvar)

        edge_logits = self.decode_edges(z, node_embs, data.edge_index, data.edge_attr)
        node_logits = self.decode_nodes(z, node_embs)

        result = {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "node_embs": node_embs,
            "edge_logits": edge_logits,
            "node_logits": node_logits,
        }

        if fully_decode:
            N = data.x.size(0)
            result["edge_logits_full"] = self.decode_full_adjacency(z, node_embs, N)

        return result

    # ── Loss ─────────────────────────────────────────────────────────────────

    def vae_loss(
        self,
        result: dict[str, torch.Tensor],
        data,
        kl_weight: float,
        edge_bce_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute L_RECON_EDGE + node_weight * L_RECON_NODE + kl_weight * L_KL.

        Edge reconstruction: label-smoothed BCE over all upper-triangular
        candidate edges, normalized by total number of candidates.

        Node reconstruction: cross-entropy over true atomic numbers, normalized
        by number of nodes.

        KL: D_KL(N(μ,σ) ‖ N(0,I)), already clipped and mean-reduced.

        Returns:
            (total_loss, edge_loss, node_loss, kl)
        """
        mu, logvar, z = result["mu"], result["logvar"], result["z"]
        node_embs = result["node_embs"]
        edge_logits = result["edge_logits"]
        node_logits = result["node_logits"]

        # ── KL divergence (clipped) ───────────────────────────────────────────
        kl = self._kl(mu, logvar)

        # ── Edge reconstruction (label-smoothed BCE over all candidates) ──────
        N = data.x.size(0)
        true_edge_index = data.edge_index

        # Positive edges (ground truth bonds).
        pos_logits = edge_logits
        # Label smoothing: push targets slightly toward 0.5 instead of {0,1}.
        pos_targets = (
            torch.ones(pos_logits.size(0), device=z.device)
            * (1.0 - self.label_smoothing)
        )

        # Negative samples: non-edges in the upper triangle.
        rows, cols = [], []
        for i in range(N):
            for j in range(i + 1, N):
                if not ((true_edge_index[0] == i) & (true_edge_index[1] == j)).any():
                    rows.append(i)
                    cols.append(j)

        if rows:
            neg_edge_index = torch.tensor([rows, cols], device=z.device)
            neg_attr = torch.ones(len(rows), 1, device=z.device)
            neg_logits = self.decode_edges(z, node_embs, neg_edge_index, neg_attr)
            neg_targets = (
                torch.zeros(neg_logits.size(0), device=z.device)
                * self.label_smoothing
            )

            all_logits = torch.cat([pos_logits, neg_logits])
            all_targets = torch.cat([pos_targets, neg_targets])
        else:
            all_logits = pos_logits
            all_targets = pos_targets

        # Normalize by total number of candidates → gradient is O(1/N²) not O(1).
        n_candidates = all_targets.size(0)
        edge_loss = F.binary_cross_entropy_with_logits(
            all_logits, all_targets, reduction="sum"
        ) / n_candidates

        # ── Node reconstruction ────────────────────────────────────────────────
        atomic_nums = data.x.view(-1).long().clamp(0, self.num_atom_types)
        n_nodes = atomic_nums.size(0)
        node_loss = F.cross_entropy(node_logits, atomic_nums, reduction="sum") / n_nodes

        total = (
            edge_bce_weight * edge_loss
            + self.node_weight * node_loss
            + kl_weight * kl
        )
        return total, edge_loss.detach(), node_loss.detach(), kl.detach()
