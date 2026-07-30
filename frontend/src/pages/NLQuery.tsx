import { NLQueryChat } from "@/components/NLQueryChat";

export function NLQuery() {
  return (
    <div className="space-y-4 max-w-3xl">
      <div>
        <h1 className="text-2xl font-bold">Natural Language Query</h1>
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          Stage 0 of the pipeline: convert free-text chemistry requests into structured queries against the Neo4j materials knowledge graph.
        </p>
      </div>
      <NLQueryChat />
    </div>
  );
}
