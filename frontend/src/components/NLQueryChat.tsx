import { useState } from "react";
import { Loader2, Send, Sparkles } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { runNLQuery } from "@/api/nlQuery";
import type { NLQueryResult } from "@/lib/types";
import { formatEnergy } from "@/lib/utils";

const EXAMPLES = [
  "low energy W-Ti-C",
  "top 5 Nb carbides",
  "W-V-Zr-Nb-C refractory",
  "stable Ti-V-W-C",
];

export function NLQueryChat() {
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<NLQueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function submit(p: string) {
    setPrompt(p);
    setLoading(true);
    setError(null);
    try {
      const r = await runNLQuery(p, 10);
      setResult(r);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card p-5">
      <div className="flex items-center gap-2 mb-3">
        <Sparkles className="h-4 w-4 text-emerald-500" />
        <h3 className="font-semibold">Stage 0 — NL to Neo4j</h3>
      </div>
      <p className="text-sm text-zinc-600 dark:text-zinc-400 mb-3">
        Describe what you want in plain English. We translate it to a Cypher query against the Materials knowledge graph.
      </p>
      <div className="flex flex-wrap gap-2 mb-3">
        {EXAMPLES.map((ex) => (
          <button key={ex} onClick={() => submit(ex)} className="chip">{ex}</button>
        ))}
      </div>
      <div className="flex gap-2">
        <textarea
          rows={2}
          className="input flex-1 resize-none"
          placeholder="e.g. low energy W-Ti-C with E_f below -1.0"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit(prompt);
          }}
        />
        <button
          className="btn-primary px-4"
          disabled={loading || !prompt.trim()}
          onClick={() => submit(prompt)}
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
        </button>
      </div>
      <p className="mt-2 text-[11px] text-zinc-500">Tip: ⌘ + Enter to submit.</p>

      <AnimatePresence>
        {error && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="mt-4 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-700 dark:text-red-300">
            {error}
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {result && (
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="mt-5 space-y-4">
            <div className="flex items-center gap-2">
              <span className="text-xs uppercase tracking-wide text-zinc-500">Method</span>
              <span className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${
                result.method === "llm_tool_use" ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300"
                                                 : "bg-zinc-500/15 text-zinc-700 dark:text-zinc-300"
              }`}>
                {result.method === "llm_tool_use" ? "LLM tool-use" : "Rule-based fallback"}
              </span>
            </div>

            <div className="rounded-lg bg-zinc-900 text-zinc-100 p-4 font-mono text-xs overflow-x-auto">
              <pre>{JSON.stringify(result.parsed_args, null, 2)}</pre>
            </div>

            <div>
              <p className="text-sm font-semibold mb-2">{result.candidates.length} candidates</p>
              <div className="space-y-1">
                {result.candidates.map((c) => (
                  <div key={c.uid} className="flex items-center justify-between rounded-lg border border-zinc-200 dark:border-zinc-800 px-3 py-2 text-sm">
                    <span className="font-mono">{c.formula}</span>
                    <span className="font-mono text-zinc-500">{formatEnergy(c.formation_energy_per_atom)} eV/atom</span>
                  </div>
                ))}
                {result.candidates.length === 0 && (
                  <p className="text-sm text-zinc-500">No candidates returned for this query.</p>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
