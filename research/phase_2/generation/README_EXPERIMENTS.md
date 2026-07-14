# Trainer separation

- `vae_trainer.py` + `data_loader.py`: frozen broad GraphVAE v3 baseline.
- `vae_proxy_finetune_trainer.py` + `proxy_data_loader.py`: Task 6B proxy
  fine-tuning only. It consumes the proxy artifacts and writes a separately
  named checkpoint; it must not overwrite `models/vae_model.pt`.

`graph_vae.py` is shared because the proxy run fine-tunes the same Objective v3
architecture rather than introducing a different model definition.

## Multi-seed generation campaign

`generation_campaign.py` runs the existing retrieval/generation/validation
pipeline once per seed, stores every run separately, then performs global
pre-CIF deduplication. Neo4j must be running before starting a campaign.

Small smoke test:

```bash
.conda/bin/python research/phase_2/generation/generation_campaign.py \
  --requirement "stable refractory tungsten carbide candidate" \
  --include-elements W,C \
  --domain-filter refractory_carbide_v1 \
  --max-energy 0.0 \
  --limit 5 \
  --seeds 42,43 \
  --n-samples-per-seed 5 \
  --global-top-k 10 \
  --vae-checkpoint research/phase_2/models/vae_model.pt \
  --gnn-checkpoint research/phase_2/models/gnn_formation_energy_grouped_v1.pt \
  --output research/phase_2/generation/output/w_c_campaign_smoke_v1
```

CPU campaign after the smoke test passes:

```bash
.conda/bin/python research/phase_2/generation/generation_campaign.py \
  --requirement "stable refractory tungsten carbide candidate" \
  --include-elements W,C \
  --domain-filter refractory_carbide_v1 \
  --max-energy 0.0 \
  --limit 10 \
  --seeds 42-46 \
  --n-samples-per-seed 100 \
  --global-top-k 50 \
  --vae-checkpoint research/phase_2/models/vae_model.pt \
  --gnn-checkpoint research/phase_2/models/gnn_formation_energy_grouped_v1.pt \
  --output research/phase_2/generation/output/w_c_campaign_v1
```

The downstream CIF builder consumes:

```text
<campaign output>/campaign_generation_manifest.csv
```

Pre-CIF dedup removes only the same reduced formula, site count, prototype and
site-substitution hypothesis. Different prototype-derived structures are kept
until CIF reconstruction and `StructureMatcher` dedup, so possible polymorphs
are not discarded by composition alone.
