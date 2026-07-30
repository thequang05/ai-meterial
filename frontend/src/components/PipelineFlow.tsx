import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronRight } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { fetchStages } from "@/api/pipeline";
import { cn } from "@/lib/utils";

export function PipelineFlow() {
  const { data: stages = [] } = useQuery({ queryKey: ["stages"], queryFn: fetchStages });
  const [active, setActive] = useState(0);
  const s = stages[active];

  return (
    <div>
      <div className="overflow-x-auto pb-2">
        <div className="flex items-stretch min-w-max gap-2">
          {stages.map((stage, i) => (
            <div key={stage.id} className="flex items-stretch">
              <button
                onClick={() => setActive(i)}
                className={cn(
                  "flex flex-col items-center justify-center rounded-xl px-5 py-3 border transition min-w-[140px]",
                  i === active
                    ? "bg-emerald-500/10 border-emerald-500/40 text-emerald-700 dark:text-emerald-300 shadow-sm"
                    : "bg-white dark:bg-zinc-900 border-zinc-200 dark:border-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-50 dark:hover:bg-zinc-800"
                )}
              >
                <span className="text-2xl">{stage.icon}</span>
                <span className="text-sm font-semibold mt-1">{stage.name}</span>
                <span className="text-[10px] uppercase tracking-wide opacity-70 mt-0.5">Stage {i}</span>
              </button>
              {i < stages.length - 1 && <ChevronRight className="self-center h-4 w-4 mx-1 text-zinc-400" />}
            </div>
          ))}
        </div>
      </div>

      <AnimatePresence mode="wait">
        {s && (
          <motion.div
            key={s.id}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
            className="mt-6 card p-6"
          >
            <div className="flex items-center gap-3 mb-3">
              <span className="text-3xl">{s.icon}</span>
              <div>
                <h3 className="text-lg font-semibold">{s.name}</h3>
                <p className="text-xs text-zinc-500 uppercase tracking-wide">{s.id}</p>
              </div>
            </div>
            <p className="text-sm text-zinc-700 dark:text-zinc-300 mb-4 leading-relaxed">{s.description}</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
              <div className="rounded-lg border border-zinc-200 dark:border-zinc-800 p-3 bg-zinc-50 dark:bg-zinc-900/50">
                <p className="text-xs uppercase tracking-wide text-zinc-500 mb-1">Input</p>
                <p className="font-mono text-zinc-800 dark:text-zinc-200">{s.input}</p>
              </div>
              <div className="rounded-lg border border-zinc-200 dark:border-zinc-800 p-3 bg-zinc-50 dark:bg-zinc-900/50">
                <p className="text-xs uppercase tracking-wide text-zinc-500 mb-1">Output</p>
                <p className="font-mono text-zinc-800 dark:text-zinc-200">{s.output}</p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
