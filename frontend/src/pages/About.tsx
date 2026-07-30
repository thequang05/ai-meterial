import { BookOpen, Database, Cpu, FlaskConical, ShieldCheck } from "lucide-react";

export function About() {
  return (
    <div className="space-y-6 max-w-3xl">
      <div>
        <h1 className="text-2xl font-bold">About this project</h1>
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          An end-to-end AI pipeline for the generative discovery of tungsten-carbide refractory ceramics.
        </p>
      </div>

      <div className="card p-6 space-y-4">
        <p className="text-sm leading-relaxed">
          This project chains four sequential stages: a GraphVAE crystal generative model,
          an audited formation-energy GNN, CHGNet structural relaxation, and an ML-only
          thermal-proxy evaluation against a Materials Project reference. The full system
          runs on a single laptop and produces content-hash-bound audit trails at every
          stage.
        </p>

        <div className="grid sm:grid-cols-2 gap-3">
          {[
            { icon: Database, title: "Neo4j Knowledge Graph", body: "Materials Project snapshot indexed for formation-energy retrieval." },
            { icon: Cpu, title: "GraphVAE + GNN", body: "Generative model + audited crystal-graph network trained on lab benchmarks." },
            { icon: FlaskConical, title: "CHGNet Relaxation", body: "Pretrained universal neural potential for fast structure optimization." },
            { icon: ShieldCheck, title: "ML Surrogate Eval", body: "Laptop-friendly thermal proxy replacing DFT when Quantum ESPRESSO is unavailable." },
          ].map(({ icon: Icon, title, body }) => (
            <div key={title} className="rounded-lg border border-zinc-200 dark:border-zinc-800 p-3">
              <div className="flex items-center gap-2 mb-1.5">
                <Icon className="h-4 w-4 text-emerald-500" />
                <p className="text-sm font-semibold">{title}</p>
              </div>
              <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">{body}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="card p-6">
        <div className="flex items-center gap-2 mb-3">
          <BookOpen className="h-4 w-4 text-emerald-500" />
          <h2 className="font-semibold">Citation</h2>
        </div>
        <pre className="text-xs font-mono bg-zinc-50 dark:bg-zinc-900/50 rounded-lg p-3 overflow-x-auto">
{`@software{ai_material_pipeline_2026,
  title  = {An End-to-End AI Pipeline for Discovery of
            Tungsten Carbide Refractory Ceramic Candidates},
  author = {Materials Discovery Team},
  year   = {2026},
  note   = {University Project Competition}
}`}
        </pre>
      </div>

      <div className="card p-6">
        <h2 className="font-semibold mb-3">Tech stack</h2>
        <div className="flex flex-wrap gap-2">
          {["Vite", "React 18", "TypeScript", "Tailwind", "TanStack Query", "Recharts", "Framer Motion", "FastAPI", "Neo4j", "PyTorch", "CHGNet", "Pymatgen"].map((t) => (
            <span key={t} className="chip">{t}</span>
          ))}
        </div>
      </div>
    </div>
  );
}
