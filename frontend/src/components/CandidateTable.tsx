import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowUpDown, Search } from "lucide-react";
import { fetchCandidates } from "@/api/campaigns";
import { StatusBadge } from "./StatusBadge";
import { formatEnergy, cn } from "@/lib/utils";
import type { Candidate } from "@/lib/types";

type SortKey = "rank" | "formula" | "ef_post" | "ef_gap_surrogate";

export function CandidateTable({ candidates }: { candidates?: Candidate[] }) {
  const navigate = useNavigate();
  const { data } = useQuery({ queryKey: ["campaigns"], queryFn: fetchCandidates });
  const rows = candidates ?? data?.candidates ?? [];
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({ key: "rank", dir: "asc" });

  const filtered = useMemo(() => {
    const q = query.toLowerCase();
    const r = rows.filter((c) => c.formula.toLowerCase().includes(q));
    r.sort((a, b) => {
      const va = a[sort.key];
      const vb = b[sort.key];
      const cmp = typeof va === "number" && typeof vb === "number" ? va - vb : String(va).localeCompare(String(vb));
      return sort.dir === "asc" ? cmp : -cmp;
    });
    return r;
  }, [rows, query, sort]);

  const headerCell = (key: SortKey, label: string, align: "left" | "right" = "left") => (
    <th
      className={cn("px-4 py-2 text-xs font-semibold uppercase tracking-wide", align === "right" ? "text-right" : "text-left")}
      onClick={() => setSort((s) => ({ key, dir: s.key === key && s.dir === "asc" ? "desc" : "asc" }))}
      style={{ cursor: "pointer" }}
    >
      <span className="inline-flex items-center gap-1">{label} <ArrowUpDown className="h-3 w-3 opacity-50" /></span>
    </th>
  );

  return (
    <div className="card overflow-hidden">
      <div className="p-4 border-b border-zinc-200 dark:border-zinc-800 flex items-center gap-3">
        <Search className="h-4 w-4 text-zinc-400" />
        <input
          className="flex-1 bg-transparent outline-none text-sm"
          placeholder="Search formula…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <span className="text-xs text-zinc-500">{filtered.length} / {rows.length}</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-zinc-50 dark:bg-zinc-900/50 text-zinc-600 dark:text-zinc-400">
            <tr>
              {headerCell("rank", "Rank")}
              {headerCell("formula", "Formula")}
              <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wide">Elements</th>
              {headerCell("ef_post", "E_f post (eV/atom)", "right")}
              {headerCell("ef_gap_surrogate", "ΔE_f surrogate", "right")}
              <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wide">Status</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((c) => (
              <tr
                key={c.id}
                className="border-t border-zinc-100 dark:border-zinc-800 hover:bg-emerald-500/5 cursor-pointer transition"
                onClick={() => navigate(`/campaigns/${c.id}`)}
              >
                <td className="px-4 py-3 text-zinc-500">#{c.rank}</td>
                <td className="px-4 py-3 font-mono font-medium">{c.formula}</td>
                <td className="px-4 py-3">
                  <div className="flex flex-wrap gap-1">
                    {c.elements.map((e) => (
                      <span key={e} className="inline-flex items-center rounded bg-zinc-100 dark:bg-zinc-800 px-1.5 py-0.5 text-xs font-mono">{e}</span>
                    ))}
                  </div>
                </td>
                <td className="px-4 py-3 text-right font-mono">{formatEnergy(c.ef_post)}</td>
                <td className="px-4 py-3 text-right font-mono">{formatEnergy(c.ef_gap_surrogate)}</td>
                <td className="px-4 py-3"><StatusBadge status={c.status} /></td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={6} className="px-4 py-10 text-center text-sm text-zinc-500">No candidates match your filter.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
