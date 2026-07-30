export type CandidateStatus =
  | "competitive_with_best_mp"
  | "above_mp_best"
  | "energy_too_high"
  | "chemistry_rejected"
  | "missing_cif"
  | "cif_unreadable"
  | "empty_structure"
  | "formula_mismatch";

export interface Candidate {
  id: string;
  rank: number;
  formula: string;
  elements: string[];
  num_atoms: number;
  ef_post: number;
  ef_gap_surrogate: number;
  ef_mp_best: number;
  status: CandidateStatus;
  cif_url?: string;
}

export interface PipelineStage {
  id: string;
  name: string;
  icon: string;
  description: string;
  input: string;
  output: string;
}

export interface NLQueryResult {
  method: "llm_tool_use" | "rule";
  parsed_args: Record<string, unknown>;
  candidates: Array<{
    uid: string;
    formula: string;
    elements: string[];
    formation_energy_per_atom: number;
  }>;
}

export interface HealthResponse {
  status: string;
  neo4j: "up" | "down";
  timestamp: string;
}

// ──── full pipeline runner ───────────────────────────────────────────
export type PipelineRunStatus = "running" | "succeeded" | "failed";

export interface PipelineRunSummary {
  id: string;
  prompt: string;
  created_at: string;
  status: PipelineRunStatus;
  duration_s: number | null;
  return_code: number | null;
  error: string | null;
  log_excerpt: string | null;
  candidate_count: number | null;
  ranked_count: number | null;
  output_dir: string;
}

export interface RankedCandidate {
  rank: number;
  candidate_id: string;
  formula: string;
  gnn_ef_post: number | null;
  mp_best_formula: string | null;
  mp_best_formation_energy_per_atom: number | null;
  ml_estimated_hull_gap_ev_per_atom: number | null;
  status: string | null;
  cif_path: string | null;
}

export interface PipelineRunResult {
  summary: PipelineRunSummary;
  parsed_args: Record<string, unknown>;
  method: string | null;
  llm_model: string | null;
  stage0_candidates: Array<{
    uid: string;
    formula: string;
    elements: string[];
    formation_energy_per_atom: number;
  }>;
  ranked: RankedCandidate[];
  log: string;
}
