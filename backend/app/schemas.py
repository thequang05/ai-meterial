"""Pydantic schemas mirroring frontend/src/lib/types.ts."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


CandidateStatus = Literal[
    "competitive_with_best_mp",
    "above_mp_best",
    "energy_too_high",
    "chemistry_rejected",
    "missing_cif",
    "cif_unreadable",
    "empty_structure",
    "formula_mismatch",
]


class Candidate(BaseModel):
    id: str
    rank: int
    formula: str
    elements: list[str]
    num_atoms: int
    ef_post: float
    ef_gap_surrogate: float
    ef_mp_best: float
    status: CandidateStatus
    cif_url: Optional[str] = None


class CandidatesResponse(BaseModel):
    campaign_id: str
    total: int
    candidates: list[Candidate]


class PipelineStage(BaseModel):
    id: str
    name: str
    icon: str
    description: str
    input: str
    output: str


class PipelineStagesResponse(BaseModel):
    stages: list[PipelineStage]


class NLQueryRequest(BaseModel):
    prompt: str
    limit: int = Field(default=10, ge=1, le=100)


class NLCandidate(BaseModel):
    uid: str
    formula: str
    elements: list[str]
    formation_energy_per_atom: float


class NLQueryResult(BaseModel):
    method: Literal["llm_tool_use", "llm", "rule"]
    llm_model: str | None = None
    parsed_args: dict = Field(default_factory=dict)
    candidates: list[NLCandidate] = Field(default_factory=list)

class HealthResponse(BaseModel):
    status: str
    neo4j: Literal["up", "down"]
    timestamp: str


class PipelineRunRequest(BaseModel):
    prompt: str = Field(min_length=3)
    limit: int = Field(default=10, ge=1, le=50)
    model: str | None = None
    skip_stage4: bool = False
    use_llm: bool = True


class PipelineRunSummary(BaseModel):
    id: str
    prompt: str
    created_at: str
    status: Literal["running", "succeeded", "failed"]
    duration_s: float | None = None
    return_code: int | None = None
    error: str | None = None
    log_excerpt: str | None = None
    candidate_count: int | None = None
    ranked_count: int | None = None
    output_dir: str


class PipelineRunResult(BaseModel):
    summary: PipelineRunSummary
    parsed_args: dict = Field(default_factory=dict)
    method: str | None = None
    llm_model: str | None = None
    stage0_candidates: list[NLCandidate] = Field(default_factory=list)
    ranked: list["RankedCandidate"] = Field(default_factory=list)
    log: str = ""


class RankedCandidate(BaseModel):
    rank: int
    candidate_id: str
    formula: str
    gnn_ef_post: float | None = None
    mp_best_formula: str | None = None
    mp_best_formation_energy_per_atom: float | None = None
    ml_estimated_hull_gap_ev_per_atom: float | None = None
    status: str | None = None
    cif_path: str | None = None


PipelineRunResult.model_rebuild()
