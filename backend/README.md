# ai-material backend

FastAPI service that powers the React dashboard in `frontend/`.

## Endpoints (all under `/api`)

| Method | Path                         | Purpose                              |
| ------ | ---------------------------- | ------------------------------------ |
| GET    | `/api/health`                | Service + Neo4j status               |
| GET    | `/api/campaigns`             | All 15 ML-evaluated candidates       |
| GET    | `/api/candidates/{id}`       | Single candidate + CIF URL           |
| GET    | `/api/pipeline/stages`       | 5-stage pipeline metadata            |
| POST   | `/api/nl-query`              | `{prompt, limit}` → parsed candidates |
| GET    | `/api/cifs/{file}.cif`       | Static CHGNet-relaxed CIFs           |

## Data source

Candidates are loaded from  
`research/phase_2/reports/ml_thermal_eval_v1/ml_thermal_evaluation.csv`
(15 rows = ML-evaluated W-C campaign). No Neo4j, no LLM, no DFT
required at runtime — pure read-only CSV.

## Run

```
.conda/bin/python -m uvicorn app.main:app --reload --port 8000
```

The Vite dev server (already configured in `frontend/vite.config.ts`)
proxies `/api` to this port.
