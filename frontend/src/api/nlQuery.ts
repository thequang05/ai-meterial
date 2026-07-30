import { api } from "./client";
import type { NLQueryResult } from "@/lib/types";

export async function runNLQuery(prompt: string, limit = 10): Promise<NLQueryResult> {
  try {
    const { data } = await api.post<NLQueryResult>("/nl-query", { prompt, limit });
    return data;
  } catch {
    const args = mockParse(prompt, limit);
    interface MockCandidate { uid: string; formula: string; elements: string[]; formation_energy_per_atom: number }
    const includeElements = (args.include_elements as string[] | undefined) ?? [];
    const maxEnergy = args.max_energy as number | undefined;
    const minEnergy = args.min_energy as number | undefined;
    const lim = (args.limit as number | undefined) ?? 10;
    const candidates: Array<{ uid: string; formula: string; elements: string[]; formation_energy_per_atom: number }> = MOCK_CANDIDATES_DB.filter((c: MockCandidate) => {
      if (includeElements.length && !includeElements.every((e) => c.elements.includes(e))) return false;
      if (maxEnergy !== undefined && c.formation_energy_per_atom > maxEnergy) return false;
      if (minEnergy !== undefined && c.formation_energy_per_atom < minEnergy) return false;
      return true;
    }).slice(0, lim);
    return { method: "rule" as const, parsed_args: args, candidates };
  }
}

function mockParse(prompt: string, defaultLimit: number): Record<string, unknown> {
  const ELEMENT_PATTERN = /\b([A-Z][a-z]?)\b/g;
  const KNOWN = new Set(["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi"]);
  const found: string[] = [];
  const seen = new Set<string>();
  for (const m of prompt.matchAll(ELEMENT_PATTERN)) {
    const s = m[1];
    if (KNOWN.has(s) && !seen.has(s)) { found.push(s); seen.add(s); }
  }
  const limitMatch = prompt.match(/(?:top|first)\s*(\d+)/i);
  const args: Record<string, unknown> = { limit: limitMatch ? Number(limitMatch[1]) : defaultLimit, order: "asc" };
  if (found.length) args.include_elements = found;
  const maxMatch = prompt.match(/(?:below|under|<)\s*(-?\d+\.?\d*)/i);
  const minMatch = prompt.match(/(?:above|>|greater than)\s*(-?\d+\.?\d*)/i);
  if (maxMatch) args.max_energy = Number(maxMatch[1]);
  if (minMatch) args.min_energy = Number(minMatch[1]);
  return args;
}

const MOCK_CANDIDATES_DB = [
  { uid: "mp-1",  formula: "WC",         elements: ["W","C"],      formation_energy_per_atom: -0.40 },
  { uid: "mp-2",  formula: "W2C",        elements: ["W","C"],      formation_energy_per_atom: -0.30 },
  { uid: "mp-3",  formula: "TiC",        elements: ["Ti","C"],     formation_energy_per_atom: -0.85 },
  { uid: "mp-4",  formula: "Ti3NbWC5",   elements: ["Ti","Nb","W","C"], formation_energy_per_atom: -0.35 },
  { uid: "mp-5",  formula: "Ti3VWC5",    elements: ["Ti","V","W","C"],  formation_energy_per_atom: -0.30 },
  { uid: "mp-6",  formula: "NbC",        elements: ["Nb","C"],     formation_energy_per_atom: -0.55 },
  { uid: "mp-7",  formula: "VC",         elements: ["V","C"],      formation_energy_per_atom: -0.45 },
  { uid: "mp-8",  formula: "ZrC",        elements: ["Zr","C"],     formation_energy_per_atom: -0.90 },
  { uid: "mp-9",  formula: "TaC",        elements: ["Ta","C"],     formation_energy_per_atom: -0.60 },
];
