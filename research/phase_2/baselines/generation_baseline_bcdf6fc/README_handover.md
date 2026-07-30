# Phase 2 — Crystal Structure Generation

## Mục tiêu

Xây dựng mô hình **Generative Model** cho vật liệu tinh thể: từ latent space $z$, decode ra cấu trúc graph mới (topology + atom types) mà formation energy nằm trong vùng mong muốn.

## Kiến trúc hiện tại

```
┌─────────────────────────────────────────────────────────────────┐
│                        Graph VAE                                 │
│  (theo Simonovsky et al. 2018 — message-passing VAE)            │
│                                                                  │
│  Encoder (CGConv × 3 layers → LayerNorm → FC)                    │
│    x ∈ R^(N×1)  (atomic numbers)                                 │
│    edge_attr ∈ R^(E×1)  (bond lengths in Å)                     │
│    ─────────────────────────────────────────────────────────────│
│    → mu, logvar ∈ R^latent_dim  (variational posterior)         │
│                                                                  │
│  Latent space Z ⊂ R^latent_dim                                  │
│    KL loss: D_KL(N(μ,σ) ‖ N(0,I)), clipped per-dim KL ≤ 1.0     │
│    KL weight ramp-up trong warmup epochs để reconstruction       │
│    học trước KL.                                                  │
│                                                                  │
│  Edge Decoder  MLP(concat([u_i, u_j, norm(e_ij), z]))            │
│    → Bernoulli logits cho từng candidate edge (i,j)            │
│    positive samples: ground-truth edges                          │
│    negative samples: non-edges trong upper triangle               │
│    label smoothing 0.05 để tránh saturated BCE gradient         │
│                                                                  │
│  Node Decoder  MLP(concat([u_i, z]))                             │
│    → Categorical logits cho atomic number tại mỗi node          │
└─────────────────────────────────────────────────────────────────┘

Tiếp theo: mô hình generation kết nối với pretrained GNN (phase_2/gnn_model.py)
để score formation energy từ decoded graph → chọn candidate tốt nhất.
```

## Các file chính

```
generation/
├── graph_vae.py      ← Kiến trúc VAE: encoder, edge decoder, node decoder
├── vae_trainer.py    ← Training loop với KL annealing, gradient clipping
├── data_loader.py    ← Dataset loader từ materials_graphs.pt
├── generator.py      ← Generation: sample z từ prior, decode graph
└── main.py           ← Orchestrator gộp Retrieval + Generation + Validation
```

## Dataset

- **Nguồn**: `research/phase_2/data/processed/materials_graphs.pt` (133,420 crystal graphs)
- **Data diagnostic** (`debug_data.py`):
  - `x` (atomic numbers): `[1, 94]` — hoàn toàn sạch
  - `edge_attr` (bond lengths): `[0.73, 5.0]` Å — hoàn toàn sạch
  - 251 graphs có >200 nodes (đã được filter bởi `max_nodes` loader)
- **Dataset hiện tại** (với `max_nodes=64, max_edges=256`): 37,954 graphs
  - train: 30,360 / val: 3,790 / test: 3,803

## Hyperparameters hiện tại

| Parameter | Giá trị | Ghi chú |
|---|---|---|
| `atom_emb_dim` | 64 | Embedding dimension per atom type |
| `hidden_dim` | 128 | Hidden dim cho decoder MLPs |
| `latent_dim` | 32 | Dimensionality of latent space Z |
| `num_gnn_layers` | 3 | Số CGConv layers trong encoder |
| `lr` | 5e-4 | Learning rate (đã giảm từ 1e-3) |
| `kl_weight` | 0.01 | KL loss weight |
| `kl_warmup` | 10 | Số epochs để ramp up kl_weight |
| `grad_clip` | 1.0 | Gradient norm clipping |
| `accumulate` | 4 | Gradient accumulation batches |
| `label_smoothing` | 0.05 | BCE label smoothing |
| `edge_logits_clip` | 10.0 | Clip logits trước sigmoid |
| `max_nodes` | 64 | Filter graphs có >64 nodes |
| `max_edges` | 256 | Filter graphs có >256 edges |

## Training loss

```
L_total = L_edge + β * L_node + α * L_KL

L_edge  = BCE_with_logits(edge_logits, labels)  — label-smoothed
L_node  = CrossEntropy(node_logits, atomic_numbers)
L_KL    = Σ_d (0.5 * (σ² + μ² - 1 - log σ²))  với per-dim KL capped tại 1.0
α       = ramp từ 0 → kl_weight trong kl_warmup epochs
β       = node_weight = 1.0 (cố định)
```

## Numerical stability (các fix đã áp dụng)

1. **LayerNorm sau mỗi CGConv**: ổn định activation scales qua 3 message-passing layers
2. **LayerNorm trước decoder MLPs**: bounded input range cho edge/node decoders
3. **EdgeAttrNorm** (learnable affine): normalize bond lengths về ~N(0,1)
4. **Edge logits clip ±10**: ngăn sigmoid saturation → `log(0)` overflow trong BCE
5. **GELU** thay ReLU: smooth hơn, không dead ReLU
6. **Edge decoder near-zero init**: sigmoid khởi đầu ở ~0.5, cân bằng gradient
7. **BCE reduction="sum" / N**: gradient magnitude không phụ thuộc graph size
8. **KL clipping**: logvar clamped [-10, 10], per-dim KL capped 1.0

## Trạng thái hiện tại

- [x] Debug data: xác nhận dữ liệu sạch 100% — NaN đến từ training dynamics
- [x] Thiết kế và code `graph_vae.py` với đầy đủ numerical stability measures
- [x] Cập nhật `vae_trainer.py`: LR 5e-4, KL warmup 10 epochs
- [x] Sửa bug: LayerNorm dim mismatch (161 vs 224) → **đã fix**
- [x] Sửa bug: `math.isfinite` NameError → **đã fix**
- [ ] **Chạy training thành công** — đang đợi confirm từ user

## Hướng tiếp theo (TODO)

### P0 — Training ổn định
- [ ] Confirm training chạy không NaN trong 5-10 epochs đầu
- [ ] Nếu vẫn NaN: giảm LR thêm xuống 1e-4, tăng `accumulate` lên 8
- [ ] Khi training ổn: train đủ epochs (50-100), save checkpoint

### P1 — Generation
- [ ] Code `generator.py`: sample z từ N(0,I), decode edges + nodes
- [ ] Post-generation filtering:
  - Bỏ decode node types cho elements không có trong training
  - Kiểm tra bond length合理性 (1.0–3.5 Å)
  - Kiểm tra stoichiometry合理性

### P2 — Scoring & Ranking
- [ ] Dùng pretrained GNN (phase_2/gnn_model.py) predict formation energy
- [ ] Score = -|ΔH_predicted - target_energy| hoặc tương tự
- [ ] Top-K candidates → validation

### P3 — Integration với main.py
- [ ] Kết nối Graph VAE generation vào `main.py` generation stage
- [ ] Streamlit/CLI interface cho user nhập requirement

## Lưu ý quan trọng cho người nhận

1. **NaN từ training dynamics, không phải data**: CGConv compound activations qua 3 layers là nguyên nhân chính. Các fix đã applied đủ để ổn định, nhưng cần theo dõi batch đầu.

2. **Edge decoder dimensionality**: `edge_in = 2*64 + 1 + 32 = 161` — LayerNorm shape phải match chính xác. Nếu đổi `atom_emb_dim` hoặc `latent_dim`, phải update cả 3 chỗ: `edge_in`, `node_in`, và LayerNorm dims.

3. **Latent space semantics**: $z$ cần encode formation energy distribution để generation có thể condition được. Nếu KL collapse (KL ≈ 0 từ epoch 5), cân nhắc:
   - Tăng `kl_weight` lên 0.05–0.1
   - Giảm `kl_warmup` xuống 3–5
   - Thêm discriminator loss giữa prior và posterior

4. **Generator chưa code**: `generator.py` đang là placeholder. Cần implement:
   - Sample $z \sim \mathcal{N}(0, I)$
   - Decode full adjacency matrix ($O(N^2)$ candidates)
   - Beam search hoặc threshold sampling cho edge decisions
   - Node type sampling từ node_logits

5. **Checkpoint format**: Model save tại `research/phase_2/models/vae_model.pt`, bao gồm `model_state_dict`, `optimizer_state_dict`, `epoch`, `best_val_loss`.

## References

- Simonovsky & Komodakis (2018). GraphVAE: Towards Generation of Small Graphs Using Variational Autoencoders. [arXiv:1802.03480](https://arxiv.org/abs/1802.03480)
- CGConv: message-passing layer từ `torch_geometric.nn.CGConv`
