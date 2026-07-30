import { cn } from "@/lib/utils";
import { statusLabel } from "@/lib/utils";

const COLORS: Record<string, string> = {
  competitive_with_best_mp: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border-emerald-500/30",
  above_mp_best:            "bg-amber-500/15   text-amber-700   dark:text-amber-300   border-amber-500/30",
  energy_too_high:          "bg-orange-500/15  text-orange-700  dark:text-orange-300  border-orange-500/30",
  chemistry_rejected:       "bg-red-500/15     text-red-700     dark:text-red-300     border-red-500/30",
  missing_cif:              "bg-zinc-500/15    text-zinc-700    dark:text-zinc-300    border-zinc-500/30",
  cif_unreadable:           "bg-zinc-500/15    text-zinc-700    dark:text-zinc-300    border-zinc-500/30",
  empty_structure:          "bg-zinc-500/15    text-zinc-700    dark:text-zinc-300    border-zinc-500/30",
  formula_mismatch:         "bg-red-500/15     text-red-700     dark:text-red-300     border-red-500/30",
};

export function StatusBadge({ status }: { status: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium whitespace-nowrap",
        COLORS[status] ?? COLORS.missing_cif
      )}
    >
      {statusLabel(status)}
    </span>
  );
}
