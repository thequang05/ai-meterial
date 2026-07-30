import { api } from "./client";
import type { Candidate, HealthResponse } from "@/lib/types";

const MOCK_CANDIDATES: Candidate[] = [
  { id: "01_camp_41a80c8b8b30", rank: 1, formula: "Ti3NbWC5", elements: ["Ti","Nb","W","C"], num_atoms: 10, ef_post: -0.352, ef_gap_surrogate: 0.291, ef_mp_best: -0.643, status: "competitive_with_best_mp" },
  { id: "02_camp_015e29be10e0", rank: 2, formula: "Ti3VWC5",  elements: ["Ti","V","W","C"],  num_atoms: 10, ef_post: -0.296, ef_gap_surrogate: 0.326, ef_mp_best: -0.622, status: "competitive_with_best_mp" },
  { id: "03_camp_59538db42ea7", rank: 3, formula: "Ti2VWC4",  elements: ["Ti","V","W","C"],  num_atoms: 8,  ef_post: -0.282, ef_gap_surrogate: 0.340, ef_mp_best: -0.622, status: "competitive_with_best_mp" },
  { id: "04_camp_5e6d485d7147", rank: 4, formula: "Zr3TiWC5", elements: ["Zr","Ti","W","C"], num_atoms: 10, ef_post: -0.263, ef_gap_surrogate: 0.387, ef_mp_best: -0.650, status: "competitive_with_best_mp" },
  { id: "05_camp_436da3bac8cc", rank: 5, formula: "Zr3TaWC5", elements: ["Zr","Ta","W","C"], num_atoms: 10, ef_post:  0.002, ef_gap_surrogate: 0.401, ef_mp_best: -0.399, status: "energy_too_high" },
  { id: "06_camp_aaaa1111", rank: 6,  formula: "W2C",        elements: ["W","C"],          num_atoms: 3,  ef_post: -0.100, ef_gap_surrogate: 0.500, ef_mp_best: -0.600, status: "missing_cif" },
  { id: "07_camp_aaaa1112", rank: 7,  formula: "WC",         elements: ["W","C"],          num_atoms: 2,  ef_post: -0.150, ef_gap_surrogate: 0.480, ef_mp_best: -0.630, status: "missing_cif" },
  { id: "08_camp_aaaa1113", rank: 8,  formula: "Nb2WC2",     elements: ["Nb","W","C"],     num_atoms: 5,  ef_post: -0.220, ef_gap_surrogate: 0.420, ef_mp_best: -0.640, status: "missing_cif" },
  { id: "09_camp_aaaa1114", rank: 9,  formula: "TiWC2",      elements: ["Ti","W","C"],     num_atoms: 4,  ef_post: -0.180, ef_gap_surrogate: 0.460, ef_mp_best: -0.640, status: "missing_cif" },
  { id: "10_camp_aaaa1115", rank: 10, formula: "V2WC2",      elements: ["V","W","C"],      num_atoms: 5,  ef_post: -0.140, ef_gap_surrogate: 0.490, ef_mp_best: -0.630, status: "missing_cif" },
  { id: "11_camp_aaaa1116", rank: 11, formula: "TaWC2",      elements: ["Ta","W","C"],     num_atoms: 4,  ef_post: -0.090, ef_gap_surrogate: 0.510, ef_mp_best: -0.600, status: "missing_cif" },
  { id: "12_camp_aaaa1117", rank: 12, formula: "Mo2WC2",     elements: ["Mo","W","C"],     num_atoms: 5,  ef_post: -0.110, ef_gap_surrogate: 0.520, ef_mp_best: -0.630, status: "missing_cif" },
  { id: "13_camp_aaaa1118", rank: 13, formula: "ReWC2",      elements: ["Re","W","C"],     num_atoms: 4,  ef_post: -0.080, ef_gap_surrogate: 0.540, ef_mp_best: -0.620, status: "missing_cif" },
  { id: "14_camp_aaaa1119", rank: 14, formula: "HfWC2",      elements: ["Hf","W","C"],     num_atoms: 4,  ef_post: -0.200, ef_gap_surrogate: 0.430, ef_mp_best: -0.630, status: "missing_cif" },
  { id: "15_camp_aaaa1120", rank: 15, formula: "CrWC2",      elements: ["Cr","W","C"],     num_atoms: 4,  ef_post: -0.130, ef_gap_surrogate: 0.470, ef_mp_best: -0.600, status: "missing_cif" },
];

export async function fetchCandidates(): Promise<{ campaign_id: string; total: number; candidates: Candidate[] }> {
  try {
    const { data } = await api.get("/campaigns");
    return data;
  } catch {
    return { campaign_id: "w_c_dft_campaign_v1", total: MOCK_CANDIDATES.length, candidates: MOCK_CANDIDATES };
  }
}

export async function fetchCandidate(id: string): Promise<Candidate | undefined> {
  const { candidates } = await fetchCandidates();
  return candidates.find((c) => c.id === id);
}

export async function fetchHealth(): Promise<HealthResponse> {
  try {
    const { data } = await api.get("/health");
    return data;
  } catch {
    return { status: "ok", neo4j: "down", timestamp: new Date().toISOString() };
  }
}
