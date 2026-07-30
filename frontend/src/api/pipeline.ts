import { api } from "./client";
import type { PipelineStage, PipelineRunResult } from "@/lib/types";

// Run a fresh end-to-end pipeline. This call BLOCKS server-side until
// the pipeline finishes (typical runtime 5-15s on a laptop with LM
// Studio running locally).
export async function runPipeline(req: {
  prompt: string;
  limit?: number;
  model?: string;
  use_llm?: boolean;
  skip_stage4?: boolean;
}): Promise<PipelineRunResult> {
  const { data } = await api.post("/pipeline/run", {
    prompt: req.prompt,
    limit: req.limit ?? 10,
    model: req.model ?? null,
    use_llm: req.use_llm ?? true,
    skip_stage4: req.skip_stage4 ?? false,
  }, { timeout: 120_000 });
  return data;
}

export async function fetchRun(runId: string): Promise<PipelineRunResult> {
  const { data } = await api.get(`/pipeline/runs/${runId}`);
  return data;
}

export async function fetchRuns(): Promise<PipelineRunResult["summary"][]> {
  const { data } = await api.get("/pipeline/runs");
  return data.runs ?? [];
}

export async function fetchStages(): Promise<PipelineStage[]> {
  try {
    const { data } = await api.get("/pipeline/stages");
    return data.stages;
  } catch {
    return [
      { id: "stage0", name: "NL → Neo4j", icon: "🧠", description: "Convert natural-language chemistry query to a Cypher search of the Materials knowledge graph.", input: "free-text prompt", output: "candidate materials JSON" },
      { id: "stage1", name: "GraphVAE",   icon: "🎲", description: "Crystal-graph variational autoencoder proposes candidate structures from the seed compositions.", input: "seed formulas", output: "candidate CIFs" },
      { id: "stage2", name: "GNN Audit",  icon: "📊", description: "Audited graph neural network predicts formation energy; checkpoint is SHA-256 hashed.", input: "candidate CIFs", output: "E_f (pre-CHGNet)" },
      { id: "stage3", name: "CHGNet",     icon: "⚛️", description: "Universal neural potential relaxes the structure and re-predicts E_f.", input: "CIFs", output: "relaxed CIFs + E_f (post)" },
      { id: "stage4", name: "ML Eval",    icon: "🔥", description: "ML-only thermal surrogate compares post-CHGNet E_f to the Materials Project best in the same subsystem.", input: "E_f (post)", output: "ΔE_f surrogate + status label" },
    ];
  }
}