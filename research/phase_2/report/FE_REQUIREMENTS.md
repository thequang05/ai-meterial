# FRD — Frontend (React) cho AI Material Discovery Pipeline

> Mục tiêu: web demo cho cuộc thi project. Hiển thị kết quả 5-stage pipeline:
> NL → Neo4j → GraphVAE → GNN → CHGNet → ML Thermal Eval.

## 1. Stack bắt buộc

- **Vite + React 18 + TypeScript**
- **Tailwind CSS** (styling)
- **shadcn/ui** (UI components — Button, Card, Table, Tabs, Badge)
- **Recharts** (charts)
- **React Router v6** (routing)
- **Zustand** (state, optional)
- **Axios** (HTTP)
- Backend: **FastAPI** (Python) serve JSON từ CSV

KHÔNG dùng: Material UI, Chakra UI, Ant Design (overkill cho demo).

## 2. Pages / Routes

```
/                    → Landing
/campaigns           → Danh sách campaigns (15 candidates)
/campaigns/:id       → Detail 1 candidate
/pipeline            → 5-stage pipeline diagram (interactive)
/nl-query            → Stage 0 chat UI (NL → Neo4j)
/about               → Project info, methodology
```

## 3. Backend API contract (FastAPI)

Base URL: `http://localhost:8000/api`

### GET /api/health
Response: `{ "status": "ok", "neo4j": "up|down", "timestamp": "..." }`

### GET /api/campaigns
Response:
```json
{
  "campaign_id": "w_c_dft_campaign_v1",
  "total": 15,
  "candidates": [
    {
      "id": "01_camp_41a80c8b8b30",
      "formula": "Ti3NbWC5",
      "elements": ["Ti", "Nb", "W", "C"],
      "stage": "stage4_done",
      "ef_post": -0.352,
      "ef_gap_surrogate": 0.291,
      "status": "competitive_with_best_mp",
      "rank": 1
    }
  ]
}
```

### GET /api/campaigns/:id
Response: full candidate detail (formula, energy, status, CIF link, all 5 stage outputs).

### POST /api/nl-query
Request:
```json
{ "prompt": "low energy W-Ti-C with E_f below -1.0", "limit": 10 }
```
Response:
```json
{
  "method": "llm_tool_use" | "rule",
  "parsed_args": { "include_elements": ["W","Ti","C"], "max_energy": -1.0 },
  "candidates": [...]
}
```

### GET /api/pipeline/stages
Response:
```json
{
  "stages": [
    { "id": "stage0", "name": "NL → Neo4j", "icon": "🧠", "description": "..." },
    { "id": "stage1", "name": "GraphVAE",   "icon": "🎲", "description": "..." },
    { "id": "stage2", "name": "GNN Audit",  "icon": "📊", "description": "..." },
    { "id": "stage3", "name": "CHGNet",     "icon": "⚛️", "description": "..." },
    { "id": "stage4", "name": "ML Eval",    "icon": "🔥", "description": "..." }
  ]
}
```

## 4. Components chính

### `<Navbar />`
- Logo + 5 menu items: Campaigns, Pipeline, NL Query, About
- Dark mode toggle (lưu localStorage)
- Health indicator (xanh = neo4j up, đỏ = down)

### `<CandidateTable />`
Props: `candidates: Candidate[]`
- Sortable columns: Rank, Formula, E_f (post), ΔE_f surrogate, Status
- Status badge màu:
  - `competitive_with_best_mp` → xanh lá
  - `above_mp_best` → vàng
  - `energy_too_high` → cam
  - `chemistry_rejected` → đỏ
  - `missing_cif` → xám
- Row click → navigate `/campaigns/:id`
- Search box filter theo formula
- Pagination 10 rows/page

### `<CandidateDetail />`
Layout 2 cột:
- Trái: thông tin (formula, elements, energies, status)
- Phải: structure viewer (3Dmol.js — render CIF, xoay được)
- Bottom: timeline 5 stage với status icon mỗi stage

### `<PipelineFlow />`
- Horizontal stepper 5 bước (shadcn Stepper)
- Click 1 step → mở panel bên dưới mô tả (input/output, code snippet, status)
- Animation khi scroll

### `<NLQueryChat />`
- Textarea + Submit button
- Loading spinner trong lúc gọi API
- Hiển thị:
  1. Parsed args (parsed args JSON)
  2. Method (LLM tool-use hay rule-based)
  3. Candidates table inline
- Example prompts (chips clickable):
  - "low energy W-Ti-C"
  - "top 5 Nb carbides"
  - "W-V-Zr-Nb-C refractory"
  - "stable Ti-V-W-C"

### `<EnergyChart />` (Recharts)
- Bar chart: candidates vs E_f post vs E_f MP-best
- Màu: xanh (GNN), cam (MP)
- Tooltip hiện formula + giá trị
- Toggle log-scale Y

### `<MetricsCards />`
4 card ở landing:
- "Candidates evaluated": 15
- "Chemistry passed": 4
- "Competitive with MP": 4
- "Pipeline stages": 5

## 5. Visual style guide

- **Color palette**:
  - Background: zinc-950 (dark), white (light)
  - Primary: emerald-500
  - Accent: blue-500
  - Warning: amber-500
  - Danger: red-500
- **Typography**: Inter (Google Font)
- **Spacing**: 4px grid
- **Border radius**: rounded-xl (12px)
- **Shadow**: shadow-lg cho cards
- **Animation**: framer-motion cho page transitions

## 6. Data fetching

- Dùng **TanStack Query** (React Query) cho caching
- Default staleTime: 30s
- Error boundary + skeleton loading

## 7. File structure

```
frontend/
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── routes.tsx
│   ├── pages/
│   │   ├── Landing.tsx
│   │   ├── Campaigns.tsx
│   │   ├── CandidateDetail.tsx
│   │   ├── Pipeline.tsx
│   │   ├── NLQuery.tsx
│   │   └── About.tsx
│   ├── components/
│   │   ├── Navbar.tsx
│   │   ├── CandidateTable.tsx
│   │   ├── EnergyChart.tsx
│   │   ├── PipelineFlow.tsx
│   │   ├── NLQueryChat.tsx
│   │   ├── MetricsCards.tsx
│   │   └── StatusBadge.tsx
│   ├── api/
│   │   ├── client.ts
│   │   ├── campaigns.ts
│   │   ├── nlQuery.ts
│   │   └── pipeline.ts
│   ├── lib/
│   │   ├── utils.ts
│   │   └── types.ts
│   └── styles/
│       └── globals.css
├── public/
├── index.html
├── package.json
├── vite.config.ts
├── tailwind.config.js
├── tsconfig.json
└── README.md
```

## 8. package.json dependencies tối thiểu

```json
{
  "dependencies": {
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "react-router-dom": "^6.26.0",
    "@tanstack/react-query": "^5.51.0",
    "axios": "^1.7.0",
    "recharts": "^2.12.0",
    "tailwindcss": "^3.4.0",
    "lucide-react": "^0.400.0",
    "framer-motion": "^11.3.0",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.5.0"
  },
  "devDependencies": {
    "vite": "^5.4.0",
    "@vitejs/plugin-react": "^4.3.0",
    "typescript": "^5.5.0",
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0"
  }
}
```

## 9. Acceptance criteria

- [ ] User mở `/` thấy 4 metric cards + 1 chart preview trong < 2s
- [ ] `/campaigns` load table 15 candidates, sort/filter hoạt động
- [ ] Click candidate → detail page render CIF 3D viewer
- [ ] `/nl-query` gửi prompt "W-Ti-C" → nhận lại candidates trong < 5s
- [ ] `/pipeline` stepper 5 bước click được, có animation
- [ ] Dark mode toggle persist qua reload
- [ ] Health indicator đỏ khi Neo4j down
- [ ] Mobile responsive (≥ 768px OK)
- [ ] Không có console error khi navigate qua 6 pages

## 10. Out of scope

- Authentication (không cần login)
- Real-time WebSocket (dùng polling 30s thay thế)
- i18n (English only)
- Accessibility full WCAG (chỉ đạt 70%)

## 11. Data source

FE đọc data từ CSV đã có:
- `research/phase_2/reports/ml_thermal_eval_v1/ml_thermal_evaluation.csv`

FastAPI backend parse CSV này → expose qua `/api/campaigns`.
KHÔNG cần trigger pipeline re-run từ FE.

## 12. Quick start (gửi kèm cho người dựng)

```bash
cd /Users/koiita/Downloads/ai-meterial
npm create vite@latest frontend -- --template react-ts
cd frontend
npm i react-router-dom @tanstack/react-query axios recharts \
       lucide-react framer-motion clsx tailwind-merge
npm i -D tailwindcss postcss autoprefixer @types/node
npx tailwindcss init -p

# Sau đó:
# 1. Setup Tailwind config (content: ["./index.html","./src/**/*.{js,ts,jsx,tsx}"])
# 2. Add Tailwind directives to src/styles/globals.css
# 3. Implement theo FRD
# 4. npm run dev → mở http://localhost:5173

# Backend FastAPI chạy song song ở port 8000:
# .conda/bin/python -m uvicorn api:app --reload --port 8000
```
