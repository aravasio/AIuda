import { BANNED_TERMS, CATEGORIES, MAX_ONE_LINER_WORDS } from "./schema.ts";

/** Structured facts given as context. The model reads them; it never restates them. */
export interface PromptContext {
  repoId: string;
  /** The Hub's own task label, supplied so the model can avoid parroting it. */
  pipelineTag: string | null;
  tags: string[];
  baseModel: string | null;
  isQuantization: boolean;
  isAdapter: boolean;
}

/**
 * The model's only job is turning a card's prose into fixed fields in plain
 * words. It is told, in as many ways as the prompt allows, that numbers are not
 * its business: they come from config.json and repo metadata, and anything
 * numeric it writes is discarded on the way out.
 */
export const PROSE_SYSTEM_PROMPT = `You explain machine-learning models to people who have never used one.

WHAT YOU ARE DOING
You are given a model's README. You turn it into a few short fields that tell a
newcomer what the model takes in, what it gives back, and what they would build
with it.

NEVER WRITE A NUMBER
Do not state parameter counts, file sizes, memory requirements, context lengths,
release dates, version numbers or benchmark scores. Those are read directly from
the repository and any number you write is thrown away. Naming another model is
fine; quoting its score is not.

WRITE FOR A NEWCOMER
- Say what goes in and what comes out, in concrete terms.
- Never use these words: ${BANNED_TERMS.join(", ")}.
- Do not repeat the Hub's task label as your explanation. A label like
  "token classification" describes the architecture and tells a reader nothing
  about what the model is for.

Good one_liner:
  "Takes an audio file and its written transcript, returns the exact second each
   word is spoken."
Bad one_liner:
  "Non-autoregressive token classification model for forced alignment across 11
   languages."

The good one says what goes in and what comes out. The bad one describes how it
was built.

not_for MATTERS MOST
not_for is what stops someone downloading the wrong thing. If there is an
obvious model people confuse this with, name it. Be specific: not "not for
everything else" but "not for transcribing audio, you need a separate speech
recognition model for that".

Reply with JSON only. No explanation around it, no markdown fences.`;

export function buildProsePrompt(card: string, context: PromptContext): string {
  const lines: string[] = [];

  lines.push(`Repository: ${context.repoId}`);
  if (context.pipelineTag !== null) {
    lines.push(
      `The Hub labels this "${context.pipelineTag}". That label describes the architecture. Do not use it as your explanation.`,
    );
  }
  if (context.tags.length > 0) {
    lines.push(`Hub tags: ${context.tags.slice(0, 20).join(", ")}`);
  }
  if (context.isQuantization && context.baseModel !== null) {
    lines.push(
      `This repository is a smaller-file version of ${context.baseModel}. Describe what the underlying model does, not the file format.`,
    );
  }
  if (context.isAdapter && context.baseModel !== null) {
    lines.push(
      `This repository is an add-on loaded on top of ${context.baseModel}. Describe what the add-on changes.`,
    );
  }

  lines.push("");
  lines.push("--- README BEGINS ---");
  lines.push(card);
  lines.push("--- README ENDS ---");
  lines.push("");
  lines.push(`Return exactly this JSON shape:

{
  "one_liner": "What goes in, what comes out, what you would build with it. ${MAX_ONE_LINER_WORDS} words maximum.",
  "use_for": ["two to four concrete things someone would actually use this for"],
  "not_for": "the most likely way someone would misuse this, and what to use instead",
  "category": ["one or more of: ${CATEGORIES.join(", ")}"],
  "deployment": {
    "where": "local" | "hosted" | "either",
    "reason": "one short sentence saying why"
  },
  "pairs_with": ["models this is designed to be used alongside"] or null,
  "supersedes_note": "say plainly if the README indicates a newer version replaces this, or if it is mainly of research interest" or null
}

Rules for category: use "base" only for a model with no instruction tuning that
cannot hold a conversation, and when you use it, use it alone. Use "world-model"
for a model that simulates an environment rather than answering questions.

Set supersedes_note to null unless the README itself points to a newer version.`);

  return lines.join("\n");
}

export const BENCHMARK_SYSTEM_PROMPT = `You read benchmark tables out of a model's README and return them as JSON.

A benchmark result is a test with a numeric score, such as MMLU 74.2 or
SWE-bench Verified 41.6.

Many tables in a README are not benchmarks. Supported languages, available
checkpoints, feature comparisons, model sizes and download links all appear as
tables and none of them are results. Skip them entirely. If a cell is a word
rather than a number, it is not a score.

Copy the numbers exactly as printed. Do not calculate, round, convert or
average anything. Do not add a benchmark that is not in the text.

Many tables compare several models side by side. For each score, record which
model it belongs to, using the column heading. If a table has only one model,
use the repository's own name.

Reply with JSON only. No explanation around it, no markdown fences.`;

export function buildBenchmarkPrompt(tables: string, repoId: string): string {
  return `Repository: ${repoId}

--- TABLES BEGIN ---
${tables}
--- TABLES END ---

Return exactly this JSON shape:

{
  "benchmarks": [
    {
      "benchmark": "the name of the test, for example MMLU or SWE-bench Verified",
      "score": "the number exactly as printed, as a string",
      "model": "which model this score belongs to, or null if the table does not say"
    }
  ]
}

If there are no benchmark results in the text, return {"benchmarks": []}.`;
}

/** Appended on a retry so the model is told precisely what was wrong. */
export function buildRetryNote(problems: string[]): string {
  return `

Your previous reply was rejected:
${problems.map((p) => `  - ${p}`).join("\n")}

Return corrected JSON. Same shape, nothing else.`;
}
