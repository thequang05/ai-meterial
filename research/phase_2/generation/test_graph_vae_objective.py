from __future__ import annotations

import tempfile
import unittest
import sys
from argparse import Namespace
from pathlib import Path

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generator import load_vae
from graph_vae import GraphVAE, TRAINING_OBJECTIVE_VERSION


def synthetic_graph() -> Data:
    x = torch.tensor([[8], [8], [14], [14], [26], [8]], dtype=torch.long)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 1, 0, 2, 1, 3, 2, 4, 3, 5, 4],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
        ],
        dtype=torch.long,
    )
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=torch.full((edge_index.size(1), 1), 2.2),
        y=torch.tensor([-2.0]),
    )
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    return data


def small_model() -> GraphVAE:
    model = GraphVAE(
        atom_emb_dim=16,
        hidden_dim=24,
        latent_dim=8,
        num_gnn_layers=2,
        context_dropout=0.4,
        kl_free_bits=0.02,
        energy_weight=1.0,
    )
    model.set_energy_stats(-1.0, 0.5)
    return model


class GraphVAEObjectiveV3Tests(unittest.TestCase):
    def test_objective_version(self) -> None:
        self.assertEqual(TRAINING_OBJECTIVE_VERSION, 3)

    def test_masked_target_invariance(self) -> None:
        model = small_model().eval()
        data = synthetic_graph()
        torch.manual_seed(17)
        first = model(data, fully_decode=False, corrupt=True)
        mask = first["node_mask"].clone()

        changed = data.clone()
        changed.x = data.x.clone()
        changed.x[mask] = torch.where(
            changed.x[mask] == 8,
            torch.tensor(14),
            torch.tensor(8),
        )
        torch.manual_seed(17)
        second = model(changed, fully_decode=False, corrupt=True)

        self.assertTrue(torch.equal(mask, second["node_mask"]))
        self.assertTrue(torch.allclose(first["node_logits"], second["node_logits"], atol=1e-7))
        self.assertTrue(torch.allclose(first["edge_logits"], second["edge_logits"], atol=1e-7))

    def test_held_out_edges_and_attributes_do_not_leak(self) -> None:
        model = small_model()
        data = synthetic_graph()
        torch.manual_seed(23)
        observed, _, candidates, attrs, targets = model._masked_edge_task(data)
        n = data.x.size(0)
        observed_keys = set(
            (
                torch.minimum(observed[0], observed[1]) * n
                + torch.maximum(observed[0], observed[1])
            ).tolist()
        )
        positive_keys = set(
            (candidates[0, targets.bool()] * n + candidates[1, targets.bool()]).tolist()
        )
        self.assertTrue(observed_keys.isdisjoint(positive_keys))
        self.assertGreater(int((targets == 0).sum()), 0)
        self.assertTrue(torch.allclose(attrs, attrs[:1].expand_as(attrs)))

    def test_node_loss_ignores_visible_targets(self) -> None:
        model = small_model().eval()
        data = synthetic_graph()
        torch.manual_seed(29)
        result = model(data, fully_decode=False, corrupt=True)
        original = model.vae_loss(result, data, kl_weight=0.01)[2]
        changed = dict(result)
        changed["node_logits"] = result["node_logits"].clone()
        changed["node_logits"][~result["node_mask"]] += 100.0 * torch.randn_like(
            changed["node_logits"][~result["node_mask"]]
        )
        modified = model.vae_loss(changed, data, kl_weight=0.01)[2]
        self.assertTrue(torch.allclose(original, modified, atol=1e-7))

    def test_free_bits_and_latent_modulation(self) -> None:
        model = small_model().eval()
        zeros = torch.zeros(1, model.latent_dim)
        self.assertEqual(float(model._kl(zeros, zeros)), 0.0)
        expected = model.latent_dim * model.kl_free_bits
        self.assertAlmostEqual(float(model._kl_objective(zeros, zeros)), expected, places=6)

        data = synthetic_graph()
        node_embs, _, _ = model.encode_node_embeddings(data)
        logits_a = model.decode_nodes(torch.zeros(1, model.latent_dim), node_embs)
        logits_b = model.decode_nodes(torch.ones(1, model.latent_dim), node_embs)
        self.assertGreater(float((logits_a - logits_b).abs().mean()), 1e-5)

    def test_finite_backward_and_energy_head_gradient(self) -> None:
        model = small_model().train()
        data = synthetic_graph()
        result = model(data, fully_decode=False, corrupt=True)
        losses = model.vae_loss(result, data, kl_weight=0.01)
        self.assertTrue(all(torch.isfinite(value) for value in losses))
        losses[0].backward()
        self.assertTrue(
            all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters())
        )
        self.assertTrue(any(parameter.grad is not None for parameter in model.energy_head.parameters()))

    def test_incompatible_checkpoint_is_rejected(self) -> None:
        args = Namespace(
            atom_emb_dim=16,
            hidden_dim=24,
            latent_dim=8,
            num_layers=2,
            kl_weight=0.02,
            node_weight=5.0,
            edge_pos_weight=5.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            torch.save({"objective_version": 2, "model": {}}, path)
            with self.assertRaisesRegex(ValueError, "incompatible GraphVAE objective"):
                load_vae(path, args, torch.device("cpu"))

    def test_collapsed_checkpoint_is_rejected(self) -> None:
        args = Namespace(
            atom_emb_dim=16,
            hidden_dim=24,
            latent_dim=8,
            num_layers=2,
            kl_weight=0.2,
            node_weight=5.0,
            edge_pos_weight=5.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collapsed.pt"
            torch.save(
                {
                    "objective_version": TRAINING_OBJECTIVE_VERSION,
                    "latent_gate_passed": False,
                    "model": {},
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "latent-usage gate"):
                load_vae(path, args, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
