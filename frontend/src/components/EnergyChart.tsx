import { useQuery } from "@tanstack/react-query";
import { fetchCandidates } from "@/api/campaigns";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from "recharts";
import { formatEnergy } from "@/lib/utils";

export function EnergyChart() {
  const { data } = useQuery({ queryKey: ["campaigns"], queryFn: fetchCandidates });
  const top = (data?.candidates ?? [])
    .filter((c) => c.status !== "missing_cif")
    .slice(0, 8)
    .map((c) => ({ formula: c.formula, "E_f (GNN)": c.ef_post, "E_f (MP best)": c.ef_mp_best }));

  return (
    <div className="card p-5">
      <div className="flex items-baseline justify-between mb-4">
        <h3 className="font-semibold">Formation Energy — Top Candidates</h3>
        <p className="text-xs text-zinc-500">CHGNet-relaxed vs. Materials Project reference (eV/atom)</p>
      </div>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={top}>
            <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="opacity-20" />
            <XAxis dataKey="formula" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip
              formatter={(v: number) => formatEnergy(v)}
              contentStyle={{ background: "rgba(24,24,27,0.95)", border: "1px solid #3f3f46", borderRadius: 8, fontSize: 12 }}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="E_f (GNN)" fill="#10b981" radius={[4, 4, 0, 0]} />
            <Bar dataKey="E_f (MP best)" fill="#f59e0b" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
