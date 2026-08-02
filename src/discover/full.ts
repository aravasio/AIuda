import { HfClient } from "../hf/client.ts";
import { resolveRepo } from "../pipeline/resolve.ts";
import { analyze, type VariantFit } from "../pipeline/analyze.ts";
import { loadConfig } from "../config.ts";
import type { MachineSpecs } from "../specs/detect.ts";
import { createProgress } from "../render/progress.ts";
import type { RegistryEntry } from "./types.ts";
import { bestBenchmark } from "./ranker.ts";

export interface FullResult extends RegistryEntry {
  fit: VariantFit;
  analysisSha: string;
  notes: string[];
}

/**
 * Runs the real HF pipeline for every entry in the registry.
 * Network-heavy on first run; cached after that.
 */
export async function fullAnalysis(
  entries: RegistryEntry[],
  specs: MachineSpecs,
  contextTokens: number,
  json: boolean,
): Promise<FullResult[]> {
  const config = loadConfig();
  const client = new HfClient();
  const results: FullResult[] = [];
  const progress = createProgress({ json });

  for (let i = 0; i < entries.length; i++) {
    const entry: RegistryEntry = entries[i]!;
    const label = `[${i + 1}/${entries.length}] ${entry.name}`;

    try {
      progress.start(label);
      const resolved = await resolveRepo(entry.hfUrl, client);
      const analysis = analyze(resolved.primary, resolved.base, specs, {
        contextTokens,
        runtimeOverheadBytes: config.runtimeOverheadBytes,
      });

      const bestFit = analysis.fits[0];
      if (bestFit === undefined) {
        progress.done("no weight variants found");
        continue;
      }

      const result: FullResult = {
        repoId: entry.repoId,
        name: entry.name,
        hfUrl: entry.hfUrl,
        sizeGb: entry.sizeGb,
        paramsTotal: entry.paramsTotal,
        paramsActive: entry.paramsActive,
        context: entry.context,
        architecture: entry.architecture,
        category: entry.category,
        tags: entry.tags,
        benchmarks: entry.benchmarks,
        note: entry.note,
        fit: bestFit,
        analysisSha: analysis.sha,
        notes: analysis.notes,
      };
      results.push(result);

      progress.done(verdictEmoji(bestFit.verdict.kind));
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      progress.done(`error: ${msg.slice(0, 60)}`);
    }
  }

  progress.stop();
  return results;
}

function verdictEmoji(kind: string): string {
  switch (kind) {
    case "runs-comfortably": return "✓";
    case "runs-tight":
    case "runs-slowly":
    case "split-possible": return "~";
    case "will-not-run": return "✗";
    default: return "?";
  }
}

export { bestBenchmark };