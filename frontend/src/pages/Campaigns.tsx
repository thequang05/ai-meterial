import { CandidateTable } from "@/components/CandidateTable";

export function Campaigns() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-bold">Campaign: w_c_dft_campaign_v1</h1>
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          15 refractory ceramic candidates evaluated through Stage 0–4.
        </p>
      </div>
      <CandidateTable />
    </div>
  );
}
