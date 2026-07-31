import { RevisionCache } from "../../cache/index.ts";
import { loadConfig } from "../../config.ts";
import { UsageError } from "../../errors.ts";
import { HfClient } from "../../hf/client.ts";
import { parseModelRef } from "../../hf/url.ts";
import { OllamaRuntime } from "../../llm/runtime.ts";
import { analyze } from "../../pipeline/analyze.ts";
import { runQuery, type QueryResult } from "../../pipeline/query.ts";
import { resolveRepo } from "../../pipeline/resolve.ts";
import { renderQuery } from "../../render/query.ts";
import { detectMachine } from "../../specs/detect.ts";
import type { GlobalFlags } from "../args.ts";

/**
 * What the cache holds for a revision. The prose is the expensive part and the
 * part that does not change between runs; the fit numbers depend on this
 * machine and on --context, so they are recomputed every time.
 */
interface CachedQuery {
  prose: QueryResult["prose"];
  benchmarks: QueryResult["benchmarks"];
  droppedClaims: QueryResult["droppedClaims"];
  model: string;
}

export async function queryCommand(positionals: string[], flags: GlobalFlags): Promise<number> {
  const target = positionals[0];
  if (target === undefined) {
    throw new UsageError(
      "query needs a model to look at.",
      "catalog query https://huggingface.co/Qwen/QwQ-32B",
    );
  }

  const config = loadConfig();
  const ref = parseModelRef(target);
  const client = new HfClient();
  const specs = detectMachine();
  const contextTokens = flags.context ?? config.defaultContextTokens;

  const resolved = await resolveRepo(ref, client);
  const analysis = analyze(resolved.primary, resolved.base, specs, {
    contextTokens,
    runtimeOverheadBytes: config.runtimeOverheadBytes,
  });
  if (resolved.baseError !== null) analysis.notes.push(resolved.baseError);

  const cache = new RevisionCache();
  const cached = flags.refresh ? null : cache.get<CachedQuery>(analysis.sha);

  let result: QueryResult;
  if (cached !== null && cached.model === config.model) {
    result = {
      analysis,
      prose: cached.prose,
      benchmarks: cached.benchmarks,
      droppedClaims: cached.droppedClaims,
      trimming: { prose: { tokens: 0, budgetTokens: 0, dropped: [] }, benchmarks: null },
      contextPlan: {
        windowTokens: config.llmContextTokens,
        outputTokens: config.responseTokenBudget,
        overheadTokens: 0,
        cardBudgetTokens: 0,
        cappedByModel: false,
      },
      notes: ["Read from the saved result for this exact revision of the repository."],
    };
  } else {
    // The runtime is checked before any work is done, and a missing one is
    // fatal: the explanation is the product, and a run that quietly drops it
    // looks complete while being useless.
    const runtime = new OllamaRuntime(config);
    result = await runQuery({
      snapshot: resolved.primary,
      base: resolved.base,
      analysis,
      runtime,
      config,
    });
    cache.set<CachedQuery>(analysis.sha, analysis.repoId, {
      prose: result.prose,
      benchmarks: result.benchmarks,
      droppedClaims: result.droppedClaims,
      model: config.model,
    });
  }

  analysis.notes.push(...result.notes);

  if (flags.json) {
    process.stdout.write(`${JSON.stringify({ ...result, machine: specs }, null, 2)}\n`);
    return 0;
  }

  process.stdout.write(renderQuery(result, specs, { technical: flags.technical }));
  return 0;
}
