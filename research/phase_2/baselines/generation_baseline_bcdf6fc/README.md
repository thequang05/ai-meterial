# Phase 2: Generative Inverse Design (Generation Stage)

Khai báo module để import:

```python
from generation import (
    run_retrieval,
    run_generation,
    run_validation,
    run_pipeline,
    GenerationConfig,
)
```

## File Structure

```
research/phase_2/generation/
├── graph_vae.py          # Graph Variational Autoencoder (encoder + decoders + loss)
├── data_loader.py         # PyG DataLoader với filter + stratified split
├── vae_trainer.py         # Training pipeline (gradient accumulation, KL annealing)
├── generator.py          # Latent-space sampler + GNN scoring
├── main.py               # Orchestrator (Retrieval → Generation → Validation)
└── __init__.py
```

## Quick Start

### 1. Train Graph VAE

```bash
cd research/phase_2/generation
python vae_trainer.py --epochs 50 --latent-dim 32 --kl-weight 0.01
```

Các tham số quan trọng:
- `--latent-dim 32` — kích thước không gian latent (mặc định)
- `--kl-weight 0.01` — trọng số KL divergence trong VAE loss
- `--max-nodes 64` — bỏ qua các graph có hơn 64 nguyên tử
- `--resume checkpoints/vae_epoch50.pt` — tiếp tục training

Checkpoint được lưu tại `research/phase_2/models/vae_model.pt`.

### 2. Generate novel materials

```bash
# Retrieval + Generation + Validation (full pipeline)
python main.py \
    --requirement "material that can withstand 1500C and is non-conductive" \
    --include-elements O \
    --max-energy -1.5 \
    --n-samples 20 \
    --top-k 10

# Retrieval only (Phase 1)
python main.py --phase 1 --include-elements O --limit 5

# Retrieval + Generation (Phase 1 + 2)
python main.py --phase 2 --include-elements O
```

### 3. Standalone generation with prototype uids

```bash
python generator.py \
    --checkpoint ../models/vae_model.pt \
    --uids MP_mp-1173034,MP_mp-1100894 \
    --interpolate \
    --n-samples 20
```

## Architecture: Graph VAE

```
Encoder (CGConv × 3):
  x → atom_embedding → conv → conv → conv → global_mean_pool
  → μ (fc_mu) + log σ² (fc_logvar) → z = μ + σ·ε

Decoders:
  Edge: MLP(z ⊕ u_i ⊕ u_j ⊕ e_ij) → Bernoulli(edge exists)
  Node: MLP(atom_emb_i ⊕ z) → categorical(atom type)

Loss: L = BCE_edges + node_weight · CE_nodes + kl_weight · D_KL
```

- **Encoder** học phân bố của crystal graphs trong không gian latent Z
- **Edge decoder** sinh topology của graph (liên kết giữa các nguyên tử)
- **Node decoder** sinh loại nguyên tử tại mỗi vị trí
- Không gian latent Z mã hóa cả cấu trúc lẫn nhiệt động học → thuận tiện cho inverse design

## Generation Flow

```
Prototype Materials (từ Neo4j)
    ↓ encode
Latent means {z₁, z₂, ..., zₖ}
    ↓ interpolate / perturb
Latent vectors {z'}
    ↓ decode
Crystal graphs candidate
    ↓ score
Pretrained GNN (formation energy prediction)
    ↓ rank
Proposed Materials (sorted by stability)
```

## Integration với Phase 1 & Phase 3

```
Phase 1 (Retrieval)     Phase 2 (Generation)      Phase 3 (Validation)
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Neo4j MCP       │    │ Graph VAE        │    │ Pretrained GNN   │
│ find_by_        │───▶│ encode prototypes│───▶│ score candidates │
│ formation_energy│    │ decode latent z  │    │ rank by stability │
└─────────────────┘    └──────────────────┘    └──────────────────┘
```

- **Phase 1** (`retrieval.py`): trả về danh sách vật liệu liên quan từ Neo4j
- **Phase 2** (Graph VAE): học phân bố của crystal graphs, sinh ứng viên mới
- **Phase 3** (`gnn_model.py`): dùng GNN đã pretrained để xác thực formation energy

## Configuration

```python
config = GenerationConfig(
    vae_checkpoint=Path("models/vae_model.pt"),
    gnn_checkpoint=Path("models/gnn_formation_energy_model.pt"),
    n_samples=20,
    interpolate=True,
    perturb_scale=0.5,
    edge_threshold=0.5,
    top_k=10,
)
```

## Output

```
research/phase_2/generation/output/
├── generation_manifest.csv   # Tất cả validated candidates
└── generation_summary.json    # Metadata + top candidates
```

Manifest CSV columns:
- `candidate_id`, `formula`, `prototype_uid`, `prototype_formula`
- `num_atoms`, `num_edges`, `gnn_formation_energy`
- `generation_method`, `latent_alpha`
