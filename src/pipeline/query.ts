import type { CatalogConfig } from "../config.ts";
import { CardTooLargeError } from "../errors.ts";
import type { RepoSnapshot } from "../hf/types.ts";
import { extractBenchmarks, extractProse } from "../llm/extract.ts";
import { buildBenchmarkPrompt, buildProsePrompt, PROSE_SYSTEM_PROMPT, type PromptContext } from "../llm/prompts.ts";
import { normaliseProse, type LabelledBenchmark, type Prose } from "../llm/schema.ts";
import { estimateTokens, looksLikeScore, trimCard, type TrimResult } from "../llm/trim.ts";
import { planContext, type ContextPlan, type LlmRuntime } from "../llm/runtime.ts";
import type { Analysis } from "./analyze.ts";
import { crossCheck, type DroppedClaim } from "./crosscheck.ts";

/** The full answer for one revision: deterministic numbers plus checked prose. */
export interface QueryResult {
  analysis: Analysis;
  prose: Prose;
  benchmarks: LabelledBenchmark[];
  /** Language-model claims thrown out because the repo's own data disagreed. */
  droppedClaims: DroppedClaim[];
  /** What the card looked like after trimming, for the technical view. */
  trimming: {
    prose: { tokens: number; budgetTokens: number; dropped: string[] };
    benchmarks: { tokens: number; budgetTokens: number } | null;
  };
  contextPlan: ContextPlan;
  /** Notes about the language-model pass itself. */
  notes: string[];
}

/**
 * Branch B of the pipeline, then the cross-check.
 *
 * The card is trimmed before the model sees it, the two extractions happen
 * separately, and every claim is checked against branch A before anything is
 * allowed into the result.
 */
export async function runQuery(input: {
  snapshot: RepoSnapshot;
  /** Base model snapshot, whose card carries the real description for a quantisation. */
  base: RepoSnapshot | null;
  analysis: Analysis;
  runtime: LlmRuntime;
  config: CatalogConfig;
}): Promise<QueryResult> {
  const { snapshot, base, analysis, runtime, config } = input;
  const notes: string[] = [];

  const status = await runtime.requireReady();

  // A quantisation's own card is usually download instructions. The description
  // belongs to the model it was made from.
  const cardSource = chooseCard(snapshot, base);
  if (cardSource.from !== snapshot.info.repoId) {
    notes.push(`Description read from the base model's card, ${cardSource.from}.`);
  }
  if (cardSource.card === null) {
    throw new CardTooLargeError(
      `${snapshot.info.repoId} has no README to read, and neither does anything it is built from. There is no description to summarise.`,
      "The repository's own page may still have useful information: https://huggingface.co/" + snapshot.info.repoId,
    );
  }

  const plan = planContext({
    requestedWindow: config.llmContextTokens,
    modelMaxContext: status.modelMaxContext,
    outputTokens: config.responseTokenBudget,
    overheadTokens: estimateTokens(PROSE_SYSTEM_PROMPT) + PROMPT_SCAFFOLD_TOKENS,
  });

  if (plan.cappedByModel) {
    notes.push(
      `${runtime.model} tops out at ${plan.windowTokens.toLocaleString("en-US")} tokens, below the ${config.llmContextTokens.toLocaleString("en-US")} configured, so the card had less room than asked for.`,
    );
  }

  const context: PromptContext = {
    repoId: snapshot.info.repoId,
    pipelineTag: snapshot.info.pipelineTag,
    tags: snapshot.info.tags,
    baseModel: analysis.classification.baseModel,
    isQuantization: analysis.classification.kind === "quantization",
    isAdapter: analysis.classification.kind === "adapter",
  };

  // Pass 1: prose. Trimming happens here, before the model, by rule.
  const trimmedProse = trimCard(cardSource.card, {
    budgetTokens: plan.cardBudgetTokens,
    pass: "prose",
  });
  if (trimmedProse.dropped.length > 0) {
    notes.push(
      `The card was too long to read whole. These sections were left out: ${trimmedProse.dropped.join(", ")}.`,
    );
  }

  const prose = await extractProse(runtime, {
    prompt: buildProsePrompt(trimmedProse.text, context),
    contextTokens: plan.windowTokens,
    maxOutputTokens: config.responseTokenBudget,
    maxRetries: config.maxSchemaRetries,
  });

  if (prose.attempts.length > 1) {
    notes.push(
      `The language model needed ${prose.attempts.length} attempts to produce a valid answer.`,
    );
  }

  // Pass 2: benchmark tables, as a separate call receiving only the tables.
  const benchmarkTrim = tryTrimBenchmarks(cardSource.card, plan.cardBudgetTokens);
  const cardBenchmarks: LabelledBenchmark[] = [];
  const rejectedRows: DroppedClaim[] = [];
  if (benchmarkTrim !== null) {
    const extracted = await extractBenchmarks(runtime, {
      prompt: buildBenchmarkPrompt(benchmarkTrim.text, snapshot.info.repoId),
      contextTokens: plan.windowTokens,
      maxOutputTokens: config.responseTokenBudget,
      maxRetries: config.maxSchemaRetries,
    });
    // A row whose score is not a number is not a benchmark, whatever the model
    // called it. Dropped by rule rather than retried: the deterministic side
    // can tell a score from a language name, and asking again would not help.
    for (const entry of extracted.value.benchmarks) {
      if (looksLikeScore(entry.score)) {
        cardBenchmarks.push({ ...entry, source: "card-table" });
      } else {
        rejectedRows.push({
          field: "benchmarks",
          claim: `${entry.benchmark}: ${entry.score}`,
          reason: "that is not a score, so it is not a benchmark result",
        });
      }
    }
  }

  // model-index numbers are structured metadata, so they are read directly and
  // never go near the language model.
  const structured: LabelledBenchmark[] = analysis.modelIndexBenchmarks.map((entry) => ({
    benchmark: [entry.task, entry.dataset, entry.metric].filter((v) => v !== null).join(" / "),
    score: String(entry.value),
    model: analysis.repoId,
    source: "model-index" as const,
  }));

  const checked = crossCheck(normaliseProse(prose.value), analysis);

  return {
    analysis,
    prose: checked.prose,
    benchmarks: [...structured, ...cardBenchmarks],
    droppedClaims: [...checked.dropped, ...rejectedRows],
    trimming: {
      prose: {
        tokens: trimmedProse.tokens,
        budgetTokens: trimmedProse.budgetTokens,
        dropped: trimmedProse.dropped,
      },
      benchmarks:
        benchmarkTrim === null
          ? null
          : { tokens: benchmarkTrim.tokens, budgetTokens: benchmarkTrim.budgetTokens },
    },
    contextPlan: plan,
    notes,
  };
}

/** Roughly what the JSON shape instructions and framing cost, on top of the system prompt. */
const PROMPT_SCAFFOLD_TOKENS = 400;

function chooseCard(
  snapshot: RepoSnapshot,
  base: RepoSnapshot | null,
): { card: string | null; from: string } {
  const own = snapshot.card;
  // A card of a few hundred characters is a download note, not a description.
  const ownIsSubstantial = own !== null && own.trim().length > 400;
  if (ownIsSubstantial || base === null || base.card === null) {
    return { card: own, from: snapshot.info.repoId };
  }
  return { card: base.card, from: base.info.repoId };
}

/**
 * A card with no tables is the normal case, not a failure, so this returns null
 * rather than throwing. The prose pass is the one that must not be skipped.
 */
function tryTrimBenchmarks(card: string, budgetTokens: number): TrimResult | null {
  try {
    return trimCard(card, { budgetTokens, pass: "benchmarks" });
  } catch {
    return null;
  }
}
