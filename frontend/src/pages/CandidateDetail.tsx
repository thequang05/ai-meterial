import { Link, useParams } from "react-router-dom";
import { ArrowLeft, Atom } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { fetchCandidate, fetchCandidates } from "@/api/campaigns";
import { fetchStages } from "@/api/pipeline";
import { StatusBadge } from "@/components/StatusBadge";
import { formatEnergy } from "@/lib/utils";

const SAMPLE_CIF = `# Generated CIF (mock)
data_Ti3NbWC5
_chemical_formula_structural 'Ti3NbWC5'
_cell_length_a    4.5123
_cell_length_b    4.5123
_cell_length_c    4.5123
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 90
loop_
_space_group_symop_operation_xyz
  'x, y, z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
  Ti1  Ti  0.000  0.000  0.000
  Ti2  Ti  0.500  0.500  0.000
  Ti3  Ti  0.500  0.000  0.500
  Nb1  Nb  0.000  0.500  0.500
  W1   W   0.250  0.250  0.250
  C1   C   0.750  0.750  0.750
  C2   C   0.250  0.750  0.250
  C3   C   0.750  0.250  0.250
  C4   C   0.250  0.250  0.750
  C5   C   0.000  0.000  0.500
`;

export function CandidateDetail() {
  const { id = "" } = useParams();
  const { data: candidate } = useQuery({ queryKey: ["candidate", id], queryFn: () => fetchCandidate(id) });
  const { data: campaign } = useQuery({ queryKey: ["campaigns"], queryFn: fetchCandidates });
  const { data: stages = [] } = useQuery({ queryKey: ["stages"], queryFn: fetchStages });

  const total = campaign?.candidates.length ?? 0;
  const completed = candidate?.status !== "missing_cif" ? 5 : 2;

  if (!candidate) {
    return (
      <div className="card p-10 text-center">
        <p className="text-zinc-500">Loading candidate…</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Link to="/campaigns" className="inline-flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100">
        <ArrowLeft className="h-4 w-4" /> Back to campaigns
      </Link>

      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 space-y-4">
          <div className="card p-5">
            <p className="text-xs uppercase tracking-wide text-zinc-500">Formula</p>
            <p className="font-mono text-3xl font-bold mt-1">{candidate.formula}</p>
            <div className="mt-3 flex flex-wrap gap-1">
              {candidate.elements.map((e) => (
                <span key={e} className="inline-flex items-center rounded bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 px-2 py-0.5 text-xs font-mono">
                  {e}
                </span>
              ))}
            </div>
            <div className="mt-3"><StatusBadge status={candidate.status} /></div>
            <p className="text-xs text-zinc-500 mt-2 font-mono break-all">{candidate.id}</p>
          </div>

          <div className="card p-5 space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-zinc-500">Rank</span>
              <span className="font-mono font-semibold">#{candidate.rank} / {total}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-zinc-500">E_f post-CHGNet</span>
              <span className="font-mono">{formatEnergy(candidate.ef_post)} eV/atom</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-zinc-500">E_f MP best</span>
              <span className="font-mono">{formatEnergy(candidate.ef_mp_best)} eV/atom</span>
            </div>
            <div className="flex justify-between text-sm border-t border-zinc-200 dark:border-zinc-800 pt-3">
              <span className="text-zinc-500">ΔE_f surrogate</span>
              <span className="font-mono font-semibold text-emerald-600 dark:text-emerald-400">
                {formatEnergy(candidate.ef_gap_surrogate)} eV/atom
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-zinc-500">Num atoms</span>
              <span className="font-mono">{candidate.num_atoms}</span>
            </div>
          </div>
        </div>

        <div className="lg:col-span-2 space-y-4">
          <div className="card p-5">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-semibold">Structure Preview</h3>
              <span className="text-xs text-zinc-500">Mock 3D viewer — replace with 3Dmol.js for live CIF</span>
            </div>
            <div className="aspect-square w-full max-w-md mx-auto rounded-xl border border-zinc-200 dark:border-zinc-800 bg-gradient-to-br from-zinc-100 to-zinc-200 dark:from-zinc-900 dark:to-zinc-800 flex items-center justify-center relative overflow-hidden">
              <div className="absolute inset-0 opacity-30" style={{
                backgroundImage: "radial-gradient(circle, currentColor 1px, transparent 1px)",
                backgroundSize: "20px 20px",
              }} />
              <div className="relative grid grid-cols-3 gap-3">
                {candidate.elements.slice(0, 6).map((e, i) => (
                  <div
                    key={i}
                    className="h-12 w-12 rounded-full flex items-center justify-center font-mono font-bold text-white shadow-lg"
                    style={{
                      backgroundColor: ELEMENT_COLOR[e] ?? "#71717a",
                      transform: `translateY(${Math.sin(i) * 8}px)`,
                    }}
                  >
                    {e}
                  </div>
                ))}
              </div>
              <div className="absolute bottom-3 left-3 flex items-center gap-1 text-xs text-zinc-500">
                <Atom className="h-3 w-3" /> Pseudo-cubic · 4.51 Å
              </div>
            </div>
          </div>

          <div className="card p-5">
            <h3 className="font-semibold mb-3">Pipeline progression — {completed}/5 stages</h3>
            <ol className="space-y-2">
              {stages.map((stage, i) => {
                const done = i < completed;
                return (
                  <li key={stage.id} className="flex items-center gap-3 rounded-lg border border-zinc-200 dark:border-zinc-800 p-3">
                    <span className={`h-7 w-7 rounded-full flex items-center justify-center text-sm font-semibold ${
                      done ? "bg-emerald-500 text-white" : "bg-zinc-200 dark:bg-zinc-800 text-zinc-500"
                    }`}>
                      {done ? "✓" : i + 1}
                    </span>
                    <span className="text-xl">{stage.icon}</span>
                    <div className="flex-1">
                      <p className="text-sm font-medium">{stage.name}</p>
                      <p className="text-xs text-zinc-500">{stage.input} → {stage.output}</p>
                    </div>
                    <span className={`text-xs font-medium ${done ? "text-emerald-600" : "text-zinc-500"}`}>
                      {done ? "Done" : "Pending"}
                    </span>
                  </li>
                );
              })}
            </ol>
          </div>

          <div className="card p-5">
            <h3 className="font-semibold mb-2">CIF (mock)</h3>
            <pre className="text-xs font-mono bg-zinc-50 dark:bg-zinc-900/50 rounded-lg p-3 overflow-x-auto max-h-72 overflow-y-auto">
              {SAMPLE_CIF}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}

const ELEMENT_COLOR: Record<string, string> = {
  W: "#3b82f6", Ti: "#a78bfa", Nb: "#10b981", Ta: "#f59e0b",
  V: "#ef4444", Zr: "#06b6d4", C: "#64748b", N: "#8b5cf6", O: "#ec4899",
};
