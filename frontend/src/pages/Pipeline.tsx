import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2, Play, RotateCcw, Terminal, ChevronRight, CircleDot } from "lucide-react";
import { runPipeline, fetchRuns, fetchRun, fetchStages } from "@/api/pipeline";
import type { PipelineRunResult, RankedCandidate } from "@/lib/types";
import { cn, formatEnergy } from "@/lib/utils";

const DEFAULT_PROMPTS = [
  "a material that can withstand temperatures of 1500 degrees Celsius and is non-conductive",
  "refractory metal carbides with low formation energy",
  "high entropy carbides containing W, Ti, and Ta",
];

export function Pipeline() {
  const qc = useQueryClient();
  const [prompt, setPrompt] = useState(DEFAULT_PROMPTS[0]);
  const [limit, setLimit] = useState(10);
  const [useLLM, setUseLLM] = useState(true);
  const [showLog, setShowLog] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  const stagesQ = useQuery({ queryKey: ["stages"], queryFn: fetchStages });
  const stages = stagesQ.data ?? [];

  const runsQ = useQuery({
    queryKey: ["pipeline-runs"],
    queryFn: fetchRuns,
    refetchInterval: (q) => (q.state.data?.some((r) => r.status === "running") ? 2000 : false),
  });

  const runMut = useMutation({
    mutationFn: () => runPipeline({ prompt, limit, use_llm: useLLM }),
    onSuccess: (data) => {
      qc.setQueryData(["pipeline-run", data.summary.id], data);
      setSelectedRunId(data.summary.id);
      qc.invalidateQueries({ queryKey: ["pipeline-runs"] });
    },
  });

  const selectedQ = useQuery({
    queryKey: ["pipeline-run", selectedRunId],
    queryFn: () => fetchRun(selectedRunId!),
    enabled: !!selectedRunId,
    refetchInterval: (q) => (q.state.data?.summary.status === "running" ? 1500 : false),
  });

  const result = selectedQ.data ?? null;
  const isRunning = runMut.isPending || result?.summary.status === "running";

  return (
    <div className="space-y-6">
      {/* header */}
      <div>
        <h1 className="text-2xl font-bold">5-Stage AI Pipeline</h1>
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          Run the full Phase 2 pipeline (NL → Neo4j → GraphVAE → GNN → CHGNet → ML Eval) end-to-end from your browser.
        </p>
      </div>

      {/* stage diagram (collapsible) */}
      <details className="card p-4 group" open={false}>
        <summary className="cursor-pointer text-sm font-medium text-zinc-700 dark:text-zinc-300 list-none flex items-center gap-2">
          <ChevronRight className="h-4 w-4 transition group-open:rotate-90" />
          Pipeline stages
        </summary>
        <div className="mt-4 overflow-x-auto pb-2">
          <div className="flex items-stretch min-w-max gap-2">
            {stages.map((s, i) => (
              <div key={s.id} className="flex items-stretch">
                <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 px-4 py-2 min-w-[140px] bg-zinc-50 dark:bg-zinc-900/50">
                  <div className="text-xl">{s.icon}</div>
                  <div className="text-xs font-semibold mt-0.5">{s.name}</div>
                  <div className="text-[10px] uppercase opacity-70">Stage {i}</div>
                </div>
                {i < stages.length - 1 && <ChevronRight className="self-center h-4 w-4 mx-1 text-zinc-400" />}
              </div>
            ))}
          </div>
        </div>
      </details>

      {/* run form */}
      <div className="card p-5 space-y-4">
        <div>
          <label className="text-xs uppercase tracking-wide text-zinc-500 font-semibold">Prompt</label>
          <textarea
            className="mt-1 w-full bg-zinc-50 dark:bg-zinc-900/60 border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/30 font-mono"
            rows={2}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="e.g. high-entropy carbides containing W and Ti"
          />
          <div className="flex flex-wrap gap-1.5 mt-2">
            {DEFAULT_PROMPTS.map((p, i) => (
              <button
                key={i}
                onClick={() => setPrompt(p)}
                className="text-[11px] rounded-full px-2.5 py-1 border border-zinc-200 dark:border-zinc-800 hover:bg-emerald-500/10 hover:border-emerald-500/40 text-zinc-600 dark:text-zinc-400"
              >
                {p.length > 48 ? p.slice(0, 45) + "…" : p}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <label className="text-xs uppercase tracking-wide text-zinc-500 font-semibold">Limit</label>
            <input
              type="number"
              min={1}
              max={50}
              value={limit}
              onChange={(e) => setLimit(Math.max(1, Math.min(50, Number(e.target.value) || 1)))}
              className="w-20 bg-zinc-50 dark:bg-zinc-900/60 border border-zinc-200 dark:border-zinc-800 rounded-md px-2 py-1 text-sm font-mono"
            />
          </div>
          <label className="flex items-center gap-2 text-sm cursor-pointer select-none">
            <input
              type="checkbox"
              checked={useLLM}
              onChange={(e) => setUseLLM(e.target.checked)}
              className="rounded border-zinc-300 text-emerald-500 focus:ring-emerald-500/30"
            />
            <span>Use LM Studio (qwen2.5-1.5b-instruct)</span>
          </label>
          <div className="flex-1" />
          <button
            onClick={() => runMut.mutate()}
            disabled={isRunning || prompt.trim().length < 3}
            className={cn(
              "inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition",
              isRunning
                ? "bg-emerald-500/30 text-emerald-700 cursor-not-allowed"
                : "bg-emerald-500 hover:bg-emerald-600 text-white shadow-sm",
            )}
          >
            {isRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            {isRunning ? "Running…" : "Run Pipeline"}
          </button>
        </div>

        {runMut.error && (
          <div className="text-sm text-red-600 dark:text-red-400 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2">
            {(runMut.error as Error).message || "Pipeline failed to start."}
          </div>
        )}
      </div>

      {/* current result */}
      <AnimatePresence mode="wait">
        {result && (
          <motion.div
            key={result.summary.id}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
            className="space-y-4"
          >
            <RunHeader result={result} />
            <ParsedArgs parsed={result.parsed_args} method={result.method} llmModel={result.llm_model} />
            <RankedTable rows={result.ranked} />
            <LogPanel log={result.log} open={showLog} setOpen={setShowLog} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* past runs */}
      <div className="card p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-500">Recent Runs</h2>
          <button
            onClick={() => runsQ.refetch()}
            className="text-xs text-zinc-500 hover:text-emerald-500 inline-flex items-center gap-1"
          >
            <RotateCcw className="h-3 w-3" /> refresh
          </button>
        </div>
        {(runsQ.data ?? []).length === 0 ? (
          <p className="text-xs text-zinc-500">No runs yet — press <em>Run Pipeline</em> above.</p>
        ) : (
          <ul className="divide-y divide-zinc-100 dark:divide-zinc-800">
            {(runsQ.data ?? []).map((r) => (
              <li
                key={r.id}
                className={cn(
                  "py-2 flex items-center gap-3 cursor-pointer rounded px-2",
                  selectedRunId === r.id ? "bg-emerald-500/10" : "hover:bg-zinc-50 dark:hover:bg-zinc-900/50",
                )}
                onClick={() => setSelectedRunId(r.id)}
              >
                <StatusDot status={r.status} />
                <div className="flex-1 min-w-0">
                  <p className="text-sm truncate font-medium">{r.prompt}</p>
                  <p className="text-[11px] text-zinc-500 font-mono">
                    {r.id} · {r.duration_s ? `${r.duration_s}s` : "—"} · cand {r.candidate_count ?? "—"} · ranked {r.ranked_count ?? "—"}
                  </p>
                </div>
                <span className="text-[11px] uppercase font-semibold text-zinc-500">{r.status}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────
function RunHeader({ result }: { result: PipelineRunResult }) {
  const s = result.summary;
  return (
    <div className="card p-4 flex flex-wrap items-center gap-4">
      <StatusDot status={s.status} big />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-semibold truncate">{s.prompt}</h2>
        </div>
        <p className="text-xs text-zinc-500 font-mono">
          run {s.id} · {new Date(s.created_at).toLocaleString()} ·{" "}
          {s.duration_s != null ? `${s.duration_s}s` : "running…"}
        </p>
      </div>
      <div className="flex gap-3 text-right">
        <Stat label="Stage 0" value={result.stage0_candidates.length} />
        <Stat label="Ranked" value={result.ranked.length} />
        <Stat label="rc" value={s.return_code ?? "—"} mono />
      </div>
    </div>
  );
}

function Stat({ label, value, mono = false }: { label: string; value: string | number; mono?: boolean }) {
  return (
    <div>
      <p className="text-[10px] uppercase tracking-wide text-zinc-500">{label}</p>
      <p className={cn("text-base font-semibold", mono && "font-mono")}>{value}</p>
    </div>
  );
}

function StatusDot({ status, big = false }: { status: string; big?: boolean }) {
  const color =
    status === "succeeded" ? "bg-emerald-500" :
    status === "failed" ? "bg-red-500" :
    status === "running" ? "bg-amber-400 animate-pulse" :
    "bg-zinc-400";
  return <CircleDot className={cn(color, "text-" + (color.split("-")[1] || "zinc") + "-500", big ? "h-5 w-5" : "h-3.5 w-3.5")} style={{ color: undefined }} />;
}

function ParsedArgs({
  parsed, method, llmModel,
}: { parsed: Record<string, unknown>; method: string | null; llmModel: string | null }) {
  if (!parsed || Object.keys(parsed).length === 0) return null;
  return (
    <div className="card p-4">
      <div className="flex items-center gap-2 mb-2">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-zinc-500">Stage 0 · Parsed</h3>
        {method && <span className="text-[10px] uppercase font-bold rounded px-1.5 py-0.5 bg-emerald-500/15 text-emerald-700 dark:text-emerald-300">{method}</span>}
        {llmModel && <span className="text-[10px] font-mono text-zinc-500">{llmModel}</span>}
      </div>
      <pre className="text-xs font-mono bg-zinc-50 dark:bg-zinc-900/60 border border-zinc-200 dark:border-zinc-800 rounded-lg p-3 overflow-x-auto">
{JSON.stringify(parsed, null, 2)}
      </pre>
    </div>
  );
}

function RankedTable({ rows }: { rows: RankedCandidate[] }) {
  if (rows.length === 0) return null;
  return (
    <div className="card overflow-hidden">
      <div className="p-4 border-b border-zinc-200 dark:border-zinc-800">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-zinc-500">Stage 4 · Ranked candidates</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-zinc-50 dark:bg-zinc-900/50 text-zinc-600 dark:text-zinc-400">
            <tr>
              <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wide">#</th>
              <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wide">Formula</th>
              <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wide">id</th>
              <th className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wide">E_f post</th>
              <th className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wide">MP best</th>
              <th className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wide">ΔE_f hull</th>
              <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wide">Status</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.candidate_id} className="border-t border-zinc-100 dark:border-zinc-800">
                <td className="px-4 py-3 text-zinc-500">#{r.rank}</td>
                <td className="px-4 py-3 font-mono font-medium">{r.formula}</td>
                <td className="px-4 py-3 font-mono text-[11px] text-zinc-500">{r.candidate_id}</td>
                <td className="px-4 py-3 text-right font-mono">{r.gnn_ef_post != null ? formatEnergy(r.gnn_ef_post) : "—"}</td>
                <td className="px-4 py-3 text-right">
                  {r.mp_best_formula
                    ? <span className="font-mono text-xs"><span className="text-zinc-500">{r.mp_best_formula}</span> <span className="ml-1 font-semibold">{r.mp_best_formation_energy_per_atom != null ? formatEnergy(r.mp_best_formation_energy_per_atom) : "—"}</span></span>
                    : "—"}
                </td>
                <td className="px-4 py-3 text-right font-mono">{r.ml_estimated_hull_gap_ev_per_atom != null ? formatEnergy(r.ml_estimated_hull_gap_ev_per_atom) : "—"}</td>
                <td className="px-4 py-3">
                  {r.status && <span className={cn(
                    "text-[10px] uppercase font-bold rounded px-1.5 py-0.5",
                    r.status === "competitive_with_best_mp" ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300" :
                    r.status === "energy_too_high" ? "bg-amber-500/15 text-amber-700 dark:text-amber-300" :
                    r.status === "missing_cif" ? "bg-zinc-500/15 text-zinc-500" :
                    "bg-zinc-500/15 text-zinc-500",
                  )}>{r.status}</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function LogPanel({ log, open, setOpen }: { log: string; open: boolean; setOpen: (b: boolean) => void }) {
  if (!log) return null;
  return (
    <div className="card overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full p-3 flex items-center gap-2 text-sm font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-zinc-900/50"
      >
        <Terminal className="h-4 w-4" />
        Pipeline log
        <span className="text-xs text-zinc-500 font-mono ml-auto">{log.length} chars</span>
      </button>
      <AnimatePresence>
        {open && (
          <motion.pre
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 320, opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="text-[11px] font-mono bg-zinc-950 text-emerald-200 dark:bg-black p-4 overflow-auto border-t border-zinc-800"
          >
{log}
          </motion.pre>
        )}
      </AnimatePresence>
    </div>
  );
}