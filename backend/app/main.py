"""FastAPI app exposing the 4 endpoints the frontend expects:

    GET  /api/health
    GET  /api/campaigns
    GET  /api/candidates/{id}
    GET  /api/pipeline/stages
    POST /api/nl-query

Data is sourced from
research/phase_2/reports/ml_thermal_eval_v1/ml_thermal_evaluation.csv
(see app/data_loader.py).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import nl as nl_module
from . import lmstudio
from . import pipeline_runner
from .data_loader import CAMPAIGN_ID, get_candidate_by_id, load_candidates
from .schemas import (
    CandidatesResponse,
    HealthResponse,
    NLQueryRequest,
    NLQueryResult,
    PipelineRunRequest,
    PipelineRunResult,
    PipelineStage,
    PipelineStagesResponse,
)

app = FastAPI(title="ai-material backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---- static CIF serving (best-effort) -------------------------------
_CIF_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "research"
    / "phase_2"
    / "dft_validation"
    / "candidate_cifs"
    / "w_c_structural_validation_v1"
)
if _CIF_DIR.is_dir():
    app.mount("/api/cifs", StaticFiles(directory=str(_CIF_DIR)), name="cifs")


# ---- health ---------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        neo4j="down",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---- candidates -----------------------------------------------------
@app.get("/api/campaigns", response_model=CandidatesResponse)
def campaigns() -> CandidatesResponse:
    candidates = load_candidates()
    return CandidatesResponse(
        campaign_id=CAMPAIGN_ID,
        total=len(candidates),
        candidates=candidates,
    )


@app.get("/api/candidates/{candidate_id}")
def candidate_detail(candidate_id: str):
    c = get_candidate_by_id(candidate_id)
    if c is None:
        raise HTTPException(status_code=404, detail="candidate not found")
    if _CIF_DIR.is_dir():
        candidate_cif = _CIF_DIR / f"{candidate_id}_chgnet_relaxed.cif"
        if candidate_cif.exists():
            c.cif_url = f"/api/cifs/{candidate_cif.name}"
    return c


# ---- pipeline stage metadata ---------------------------------------
@app.get("/api/pipeline/stages", response_model=PipelineStagesResponse)
def pipeline_stages() -> PipelineStagesResponse:
    stages = [
        PipelineStage(
            id="stage0",
            name="NL → Neo4j",
            icon="🧠",
            description="Convert natural-language chemistry query to a Cypher search of the Materials knowledge graph.",
            input="free-text prompt",
            output="candidate materials JSON",
        ),
        PipelineStage(
            id="stage1",
            name="GraphVAE",
            icon="🎲",
            description="Crystal-graph variational autoencoder proposes candidate structures from the seed compositions.",
            input="seed formulas",
            output="candidate CIFs",
        ),
        PipelineStage(
            id="stage2",
            name="GNN Audit",
            icon="📊",
            description="Audited graph neural network predicts formation energy; checkpoint is SHA-256 hashed.",
            input="candidate CIFs",
            output="E_f (pre-CHGNet)",
        ),
        PipelineStage(
            id="stage3",
            name="CHGNet",
            icon="⚛️",
            description="Universal neural potential relaxes the structure and re-predicts E_f.",
            input="CIFs",
            output="relaxed CIFs + E_f (post)",
        ),
        PipelineStage(
            id="stage4",
            name="ML Eval",
            icon="🔥",
            description="ML-only thermal surrogate compares post-CHGNet E_f to the Materials Project best in the same subsystem.",
            input="E_f (post)",
            output="ΔE_f surrogate + status label",
        ),
    ]
    return PipelineStagesResponse(stages=stages)


# ---- natural-language query ----------------------------------------
@app.post("/api/nl-query", response_model=NLQueryResult)
def nl_query(req: NLQueryRequest) -> NLQueryResult:
    parsed: dict | None = None
    method = "rule"
    llm_model: str | None = None

    # try the local LLM first; fall back to rules if anything fails
    if lmstudio.is_available():
        llm_model = lmstudio.DEFAULT_MODEL
        parsed = lmstudio.parse_with_llm(req.prompt)
        if parsed is not None:
            method = "llm"
        # else: silently degrade to rules below

    if parsed is None:
        parsed = nl_module.parse_prompt(req.prompt, default_limit=req.limit)
        if parsed.get("limit") != req.limit:
            parsed["limit"] = req.limit

    # caller-requested limit always wins as an upper bound
    try:
        llm_limit = int(parsed.get("limit", req.limit))
        parsed["limit"] = min(llm_limit, req.limit)
    except (TypeError, ValueError):
        parsed["limit"] = req.limit

    candidates = nl_module.filter_candidates(parsed)
    return NLQueryResult(
        method=method,
        llm_model=llm_model,
        parsed_args=parsed,
        candidates=candidates,
    )


# ---- full pipeline runner --------------------------------------------
@app.post("/api/pipeline/run", response_model=PipelineRunResult)
def pipeline_run(req: PipelineRunRequest) -> PipelineRunResult:
    state = pipeline_runner.start_run(
        prompt=req.prompt,
        limit=req.limit,
        model=req.model,
        use_llm=req.use_llm,
        skip_stage4=req.skip_stage4,
    )
    return _build_result(state)


@app.get("/api/pipeline/runs/{run_id}", response_model=PipelineRunResult)
def pipeline_get(run_id: str) -> PipelineRunResult:
    state = pipeline_runner.get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
    return _build_result(state)


def _build_result(state) -> PipelineRunResult:
    from .schemas import PipelineRunSummary, RankedCandidate
    s = pipeline_runner.summary(state)
    return PipelineRunResult(
        summary=PipelineRunSummary(**s),
        parsed_args=state.parsed_args,
        method=state.method,
        llm_model=state.llm_model,
        stage0_candidates=state.stage0_candidates,
        ranked=[RankedCandidate(**r) for r in state.ranked],
        log=state.log,
    )


@app.get("/api/pipeline/runs")
def pipeline_list() -> dict:
    return {"runs": pipeline_runner.list_runs()}
