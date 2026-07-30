import { Link } from "react-router-dom";
import { ArrowRight, Sparkles } from "lucide-react";
import { MetricsCards } from "@/components/MetricsCards";
import { EnergyChart } from "@/components/EnergyChart";
import { motion } from "framer-motion";

export function Landing() {
  return (
    <div className="space-y-10">
      <motion.section initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
        <div className="flex items-center gap-2 mb-3">
          <Sparkles className="h-4 w-4 text-emerald-500" />
          <span className="text-xs uppercase tracking-wide font-semibold text-emerald-600 dark:text-emerald-400">
            End-to-end AI pipeline · v1
          </span>
        </div>
        <h1 className="text-4xl sm:text-5xl font-bold tracking-tight max-w-3xl">
          Discovering refractory ceramics with a 5-stage AI pipeline.
        </h1>
        <p className="mt-4 text-zinc-600 dark:text-zinc-400 max-w-2xl">
          Natural-language query → Neo4j knowledge graph → GraphVAE generation →
          audited GNN → CHGNet relaxation → ML-only thermal surrogate. Fully
          reproducible on a single laptop.
        </p>
        <div className="mt-6 flex gap-3">
          <Link to="/campaigns" className="btn-primary">
            Browse 15 candidates <ArrowRight className="ml-1.5 h-4 w-4" />
          </Link>
          <Link to="/nl-query" className="btn-ghost">Try NL Query</Link>
        </div>
      </motion.section>

      <section>
        <MetricsCards />
      </section>

      <section>
        <EnergyChart />
      </section>

      <section className="grid md:grid-cols-3 gap-4">
        {[
          { title: "W–C Refractory Focus", body: "Tungsten carbide is a benchmark material for ultra-high-temperature applications. The campaign explores ternary compositions with Ti, V, Nb, Zr, and Ta." },
          { title: "Content-Hash Audit", body: "Every GNN checkpoint, Materials CSV, and CIF input is SHA-256 hashed at runtime. The pipeline produces reproducible audit trails." },
          { title: "Laptop-Ready", body: "When Quantum ESPRESSO is unavailable, the final stage runs as an ML-only surrogate that meaningfully orders candidates against MP references." },
        ].map((b) => (
          <div key={b.title} className="card p-5">
            <h3 className="font-semibold mb-2">{b.title}</h3>
            <p className="text-sm text-zinc-600 dark:text-zinc-400 leading-relaxed">{b.body}</p>
          </div>
        ))}
      </section>
    </div>
  );
}
