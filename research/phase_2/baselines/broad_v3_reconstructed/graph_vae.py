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

Training loss (objective v3):
  L = L_RECON_EDGE + β * L_RECON_NODE + α * L_KL_FREE_BITS
      + γ * L_ENERGY_AUX

Numerical stability measures:
  - LayerNorm after each CGConv layer → stable activation scales
  - LayerNorm on concatenated decoder inputs → bounded logits range
  - Edge attr normalization: (x - mean) / std via learnable affine transform
  - KL clipping: logvar clamped to [-10, 10], per-dim KL capped at 1.0
  - Edge logits clipped to [-10, 10] before sigmoid → no log(0) overflow
  - Focal loss for edges (handles extreme class imbalance)
  - Class-frequency weighting for node prediction
  - Free-bits KL objective + cyclical annealing (scheduled by trainer)
  - FiLM latent modulation + context dropout in the node decoder
  - Auxiliary formation-energy prediction from μ to keep z informative
  - Temperature-controlled sampling instead of argmax for generation
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_mean_pool
from typing import Tuple, Optional

TRAINING_OBJECTIVE_VERSION = 3


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
        self.shift = nn.Parameter(torch.zeros(dim) + init_mean)
        self.scale = nn.Parameter(torch.ones(dim) * init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.shift) / (self.scale + self.eps)


# ── Graph VAE ──────────────────────────────────────────────────────────────────

class GraphVAE(nn.Module):
    """
    Full VAE: encoder + edge decoder + node decoder.

    Training uses denoising reconstruction: atom labels and edges are hidden
    before encoding, then only the hidden targets contribute reconstruction loss.

    Decoder input dimensions:
      edge_in = 2*atom_emb_dim (i_emb + j_emb) + 1 (ea_norm) + latent_dim (z)
    """

    def __init__(
        self,
        atom_emb_dim: int = 64,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_gnn_layers: int = 3,
        kl_weight: float = 0.1,
        node_weight: float = 5.0,
        num_atom_types: int = 94,
        max_kl: float = 1.0,
        edge_logits_clip: float = 10.0,
        label_smoothing: float = 0.05,
        focal_gamma: float = 2.0,
        edge_pos_weight: float = 5.0,
        node_class_weights: Optional[torch.Tensor] = None,
        node_mask_rate: float = 0.30,
        edge_mask_rate: float = 0.20,
        negative_edge_ratio: float = 1.0,
        context_dropout: float = 0.40,
        kl_free_bits: float = 0.02,
        energy_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.node_weight = node_weight
        self.num_atom_types = num_atom_types
        self.max_kl = max_kl
        self.edge_logits_clip = edge_logits_clip
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.edge_pos_weight = edge_pos_weight
        self.node_mask_rate = node_mask_rate
        self.edge_mask_rate = edge_mask_rate
        self.negative_edge_ratio = negative_edge_ratio
        self.context_dropout = context_dropout
        self.kl_free_bits = kl_free_bits
        self.energy_weight = energy_weight
        self.LOGVAR_CLIP = 10.0

        # Class weights for node prediction (frequency-based, set on GPU before training).
        self.register_buffer("node_class_weights", node_class_weights if node_class_weights is not None else torch.ones(num_atom_types + 1))
        self.register_buffer("energy_mean", torch.tensor(0.0))
        self.register_buffer("energy_std", torch.tensor(1.0))

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
        energy_hidden = max(32, hidden_dim // 2)
        self.energy_head = nn.Sequential(
            nn.Linear(latent_dim, energy_hidden),
            nn.GELU(),
            nn.Linear(energy_hidden, 1),
        )

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
        self.node_film = nn.Linear(latent_dim, 2 * atom_emb_dim)
        self.node_context_ln = nn.LayerNorm(atom_emb_dim)
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

    def _kl_per_dim(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Return bounded KL per latent dimension."""
        logvar_clamped = logvar.clamp(-self.LOGVAR_CLIP, self.LOGVAR_CLIP)
        kl_per_dim = 0.5 * (
            logvar_clamped.exp() + mu.pow(2) - 1 - logvar_clamped
        )
        return kl_per_dim.clamp(max=self.max_kl)

    def _kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Raw KL used for collapse monitoring."""
        return self._kl_per_dim(mu, logvar).sum(dim=-1).mean()

    def _kl_objective(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Free-bits KL: dimensions below the allowance receive no KL gradient."""
        kl_per_dim = self._kl_per_dim(mu, logvar)
        if self.kl_free_bits <= 0:
            return kl_per_dim.sum(dim=-1).mean()
        return kl_per_dim.clamp(min=self.kl_free_bits).sum(dim=-1).mean()

    def set_energy_stats(self, mean: float, std: float) -> None:
        """Set train-split-only normalization for the auxiliary energy target."""
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid energy normalization: mean={mean}, std={std}")
        self.energy_mean.fill_(float(mean))
        self.energy_std.fill_(float(std))

    # ── Encoder ───────────────────────────────────────────────────────────────

    def encode_node_embeddings(
        self,
        data,
        atomic_nums_override: Optional[torch.Tensor] = None,
        edge_index_override: Optional[torch.Tensor] = None,
        edge_attr_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a possibly corrupted graph without reading held-out targets."""
        atomic_nums = (
            atomic_nums_override
            if atomic_nums_override is not None
            else data.x.view(-1).long()
        ).clamp(0, self.num_atom_types)
        edge_index = edge_index_override if edge_index_override is not None else data.edge_index
        edge_attr = edge_attr_override if edge_attr_override is not None else data.edge_attr

        x = self.atom_embedding(atomic_nums)
        ea_norm = self.edge_norm(edge_attr)
        for conv, ln in zip(self.convs, self.layer_norms):
            x_new = conv(x, edge_index, ea_norm)
            x = ln(x_new + x)

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph_emb = global_mean_pool(x, batch)
        return x, self.fc_mu(graph_emb), self.fc_logvar(graph_emb)

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
        _, mu, logvar = self.encode_node_embeddings(data)
        return mu, logvar

    def _masked_edge_task(self, data):
        """Remove positive edges from the encoder and build a balanced link task."""
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        n = data.x.size(0)
        device = edge_index.device

        if n < 2 or edge_index.numel() == 0:
            empty_index = edge_index.new_empty((2, 0))
            empty_attr = edge_attr.new_empty((0, edge_attr.size(-1)))
            return edge_index, edge_attr, empty_index, empty_attr, edge_attr.new_empty(0)

        lo = torch.minimum(edge_index[0], edge_index[1])
        hi = torch.maximum(edge_index[0], edge_index[1])
        non_self = lo != hi
        positive_keys = torch.unique(lo[non_self] * n + hi[non_self])
        if positive_keys.numel() == 0:
            empty_index = edge_index.new_empty((2, 0))
            empty_attr = edge_attr.new_empty((0, edge_attr.size(-1)))
            return edge_index, edge_attr, empty_index, empty_attr, edge_attr.new_empty(0)

        n_holdout = max(1, int(round(positive_keys.numel() * self.edge_mask_rate)))
        heldout_keys = positive_keys[
            torch.randperm(positive_keys.numel(), device=device)[:n_holdout]
        ]

        # Remove all directed/periodic occurrences of each held-out pair.
        directed_keys = lo * n + hi
        keep = ~torch.isin(directed_keys, heldout_keys)
        observed_edge_index = edge_index[:, keep]
        observed_edge_attr = edge_attr[keep]

        pos_i = torch.div(heldout_keys, n, rounding_mode="floor")
        pos_j = heldout_keys.remainder(n)
        pos_index = torch.stack([pos_i, pos_j])

        all_pairs = torch.triu_indices(n, n, offset=1, device=device)
        all_keys = all_pairs[0] * n + all_pairs[1]
        negative_pairs = all_pairs[:, ~torch.isin(all_keys, positive_keys)]
        n_neg = min(
            negative_pairs.size(1),
            max(1, int(round(n_holdout * self.negative_edge_ratio))),
        )
        if n_neg:
            negative_pairs = negative_pairs[:, torch.randperm(
                negative_pairs.size(1), device=device
            )[:n_neg]]
        else:
            negative_pairs = edge_index.new_empty((2, 0))

        candidate_index = torch.cat([pos_index, negative_pairs], dim=1)
        targets = torch.cat([
            torch.ones(pos_index.size(1), device=device),
            torch.zeros(negative_pairs.size(1), device=device),
        ])

        # Both classes receive the same neutral attribute, so bond length cannot
        # reveal whether a candidate is a positive or negative edge.
        neutral = self.edge_norm.shift.detach().view(1, -1)
        candidate_attr = neutral.expand(candidate_index.size(1), -1).clone()
        return observed_edge_index, observed_edge_attr, candidate_index, candidate_attr, targets

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

    def decode_nodes(self, z: torch.Tensor, node_embs: torch.Tensor) -> torch.Tensor:
        """
        Return node logits for loss computation or sampling.
        Called directly by generator.py during inference.
        """
        N = node_embs.size(0)
        B = z.size(0)
        z_expanded = z.expand(N, -1) if B == 1 else z.repeat(N, 1)
        context = F.dropout(
            node_embs,
            p=self.context_dropout,
            training=self.training,
        )
        gamma, beta = self.node_film(z_expanded).chunk(2, dim=-1)
        context = self.node_context_ln(
            context * (1.0 + torch.tanh(gamma)) + beta
        )
        node_in = self.node_ln(torch.cat([context, z_expanded], dim=-1))
        return self.node_decoder(node_in)

    # ── Full forward ──────────────────────────────────────────────────────────

    def forward(
        self, data, fully_decode: bool = True, corrupt: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Run full VAE forward pass (used by trainer for loss computation).

        For generation, use decode_nodes() + decode_full_adjacency() directly
        in generator.py to get full control over sampling temperature.
        """
        atomic_nums = data.x.view(-1).long().clamp(0, self.num_atom_types)
        node_mask = torch.ones_like(atomic_nums, dtype=torch.bool)
        encoder_atomic_nums = atomic_nums
        observed_edge_index = data.edge_index
        observed_edge_attr = data.edge_attr
        candidate_index = data.edge_index
        candidate_attr = data.edge_attr
        edge_targets = torch.ones(data.edge_index.size(1), device=atomic_nums.device)

        if corrupt:
            node_mask = (
                torch.rand(atomic_nums.size(0), device=atomic_nums.device)
                < self.node_mask_rate
            )
            if not node_mask.any():
                idx = torch.randint(atomic_nums.size(0), (1,), device=atomic_nums.device)
                node_mask[idx] = True
            encoder_atomic_nums = atomic_nums.clone()
            encoder_atomic_nums[node_mask] = 0  # reserved MASK token
            (
                observed_edge_index,
                observed_edge_attr,
                candidate_index,
                candidate_attr,
                edge_targets,
            ) = self._masked_edge_task(data)

        node_embs, mu, logvar = self.encode_node_embeddings(
            data,
            atomic_nums_override=encoder_atomic_nums,
            edge_index_override=observed_edge_index,
            edge_attr_override=observed_edge_attr,
        )
        z = self.reparameterize(mu, logvar)

        edge_logits = self.decode_edges(z, node_embs, candidate_index, candidate_attr)
        node_logits = self.decode_nodes(z, node_embs)
        energy_pred = self.energy_head(mu).view(-1)

        result = {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "node_embs": node_embs,
            "edge_logits": edge_logits,
            "node_logits": node_logits,
            "energy_pred": energy_pred,
            "node_mask": node_mask,
            "edge_targets": edge_targets,
            "candidate_edge_index": candidate_index,
        }

        if fully_decode:
            N = data.x.size(0)
            result["edge_logits_full"] = self.decode_full_adjacency(z, node_embs, N)

        return result

    # ── Loss ─────────────────────────────────────────────────────────────────

    def focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pos_weight: float = 1.0,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """
        Focal loss for handling extreme class imbalance in edge prediction.

        FL(p) = -α(1-p)^γ log(p)   for positive class
        FL(p) = -(1-α)p^γ log(1-p)  for negative class

        gamma > 0 reduces the loss contribution from well-classified examples,
        focusing training on hard examples. pos_weight upweights the rare positive class.
        """
        probs = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        modulating = (1 - p_t).pow(gamma)
        alpha_t = pos_weight * targets + (1 - targets)
        return (modulating * alpha_t * ce).sum() / logits.size(0)

    def vae_loss(
        self,
        result: dict[str, torch.Tensor],
        data,
        kl_weight: float,
        edge_bce_weight: float = 1.0,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Compute reconstruction + free-bits KL + auxiliary formation-energy loss.

        Edge: focal loss for class imbalance (sparse graph topology).
        Node: class-weighted cross-entropy (O=39%, Fe=0.9%, Ca=0.3%).
        KL: raw KL is reported; free-bits KL is optimized.
        Energy: SmoothL1 on train-normalized formation_energy_per_atom.
        """
        mu, logvar = result["mu"], result["logvar"]
        edge_logits = result["edge_logits"]
        node_logits = result["node_logits"]

        kl_raw = self._kl(mu, logvar)
        kl_objective = self._kl_objective(mu, logvar)

        # ── Edge: focal loss ────────────────────────────────────────────────
        edge_targets = result["edge_targets"]
        if edge_logits.numel():
            edge_loss = self.focal_loss(
                edge_logits,
                edge_targets,
                pos_weight=self.edge_pos_weight,
                gamma=self.focal_gamma,
            )
        else:
            edge_loss = node_logits.sum() * 0.0

        # ── Node: class-weighted cross-entropy ────────────────────────────
        atomic_nums = data.x.view(-1).long().clamp(0, self.num_atom_types)
        node_mask = result["node_mask"]
        node_loss = F.cross_entropy(
            node_logits[node_mask], atomic_nums[node_mask],
            weight=self.node_class_weights.to(node_logits.device),
            reduction="mean",
        )

        energy_target = (
            data.y.view(-1).float() - self.energy_mean
        ) / self.energy_std.clamp_min(1e-6)
        energy_loss = F.smooth_l1_loss(
            result["energy_pred"], energy_target, reduction="mean"
        )

        total = (
            edge_bce_weight * edge_loss
            + self.node_weight * node_loss
            + kl_weight * kl_objective
            + self.energy_weight * energy_loss
        )
        return (
            total,
            edge_loss.detach(),
            node_loss.detach(),
            kl_raw.detach(),
            kl_objective.detach(),
            energy_loss.detach(),
        )
