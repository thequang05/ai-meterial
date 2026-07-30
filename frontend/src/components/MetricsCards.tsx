import { useQuery } from "@tanstack/react-query";
import { fetchCandidates } from "@/api/campaigns";
import { Beaker, FlaskConical, Flame, Layers } from "lucide-react";
import { motion } from "framer-motion";

const cards = [
  { icon: FlaskConical, label: "Candidates evaluated", key: "total",      color: "text-blue-500"   },
  { icon: Beaker,       label: "Chemistry passed",     key: "chemistry",  color: "text-emerald-500" },
  { icon: Flame,        label: "Competitive with MP",  key: "competitive",color: "text-amber-500"   },
  { icon: Layers,       label: "Pipeline stages",      key: "stages",     color: "text-purple-500"  },
];

export function MetricsCards() {
  const { data } = useQuery({ queryKey: ["campaigns"], queryFn: fetchCandidates });
  const total = data?.total ?? 0;
  const chemistry = data?.candidates.filter((c) => c.status !== "chemistry_rejected" && c.status !== "missing_cif").length ?? 0;
  const competitive = data?.candidates.filter((c) => c.status === "competitive_with_best_mp").length ?? 0;
  const values: Record<string, number> = { total, chemistry, competitive, stages: 5 };

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map(({ icon: Icon, label, key, color }, i) => (
        <motion.div
          key={label}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.05 }}
          className="card p-5"
        >
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs font-medium text-zinc-500 dark:text-zinc-400 uppercase tracking-wide">{label}</p>
              <p className="mt-2 text-3xl font-bold">{values[key]}</p>
            </div>
            <Icon className={`h-6 w-6 ${color}`} />
          </div>
        </motion.div>
      ))}
    </div>
  );
}
