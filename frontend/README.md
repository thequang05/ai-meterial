# AI Material Discovery — Frontend

React + TypeScript + Vite + Tailwind UI for the AI Material Discovery pipeline.

## Quick start

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Build

```bash
npm run build
npm run preview
```

## Stack

- Vite + React 18 + TypeScript
- Tailwind CSS (dark mode by default)
- TanStack Query for data fetching
- Recharts for visualizations
- Framer Motion for transitions
- Lucide React icons
- React Router v6

## Structure

```
src/
├── main.tsx           # entry + providers
├── App.tsx            # routes
├── pages/             # Landing, Campaigns, CandidateDetail, Pipeline, NLQuery, About
├── components/        # Layout, MetricsCards, EnergyChart, CandidateTable, ...
├── api/               # axios client + fetchers
└── lib/               # utils, types
```

## Backend

By default the dev server proxies `/api/*` to `http://localhost:8000`.
See `research/phase_2/report/FE_REQUIREMENTS.md` for the full API contract.

If no backend is running, the UI falls back to mock data.

## Data source

Reads from `research/phase_2/reports/ml_thermal_eval_v1/ml_thermal_evaluation.csv`.
