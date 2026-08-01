import type { z } from "zod";
import { InvalidLlmOutputError, TruncatedReplyError } from "../errors.ts";
import type { GenerateRequest, LlmRuntime } from "./runtime.ts";
import {
  BENCHMARK_SYSTEM_PROMPT,
  buildRetryNote,
  EXAMPLE_PHRASES,
  PROSE_SYSTEM_PROMPT,
} from "./prompts.ts";
import {
  BenchmarksSchema,
  looksLikeInstruction,
  ProseSchema,
  validateCategories,
  type Category,
  type ExtractedBenchmark,
  type Prose,
} from "./schema.ts";

/** Recorded on the attempt so callers can tell a cut-off reply from a bad one. */
export const TRUNCATED_PROBLEM = "the reply was cut off before it finished";

export interface ExtractionAttempt {
  attempt: number;
  problems: string[];
  raw: string;
}

export interface ExtractionResult<T> {
  value: T;
  attempts: ExtractionAttempt[];
}

export interface ExtractOptions {
  system: string;
  prompt: string;
  contextTokens: number;
  maxOutputTokens: number;
  maxRetries: number;
  /**
   * Told when an attempt starts and as its reply arrives. A retry is most of
   * the wait on a slow machine, so a caller showing progress has to be able to
   * say that it is the second attempt rather than the first still running.
   */
  onProgress?: (event: ExtractProgress) => void;
}

export interface ExtractProgress {
  attempt: number;
  /** Attempts allowed in total, so a caller can say "2 of 3". */
  ofAttempts: number;
  /** Characters received so far in this attempt. */
  characters: number;
}

/**
 * Asks the runtime for JSON and validates it before anything is allowed near
 * the renderer. A malformed reply is retried with the specific complaint
 * attached, never patched up and never rendered.
 */
export async function extract<T>(
  runtime: LlmRuntime,
  schema: z.ZodType<T>,
  options: ExtractOptions,
  /** Extra rules that the schema alone cannot express. */
  furtherChecks: (value: T) => string[] = () => [],
): Promise<ExtractionResult<T>> {
  const attempts: ExtractionAttempt[] = [];
  let prompt = options.prompt;

  const ofAttempts = options.maxRetries + 1;

  for (let attempt = 1; attempt <= ofAttempts; attempt += 1) {
    let characters = 0;
    options.onProgress?.({ attempt, ofAttempts, characters });

    const request: GenerateRequest = {
      system: options.system,
      prompt,
      contextTokens: options.contextTokens,
      maxOutputTokens: options.maxOutputTokens,
      ...(options.onProgress === undefined
        ? {}
        : {
            onFragment: (text: string) => {
              characters += text.length;
              options.onProgress?.({ attempt, ofAttempts, characters });
            },
          }),
    };
    const { text: raw, truncated } = await runtime.generate(request);

    // A cut-off reply is not a malformed one. Reporting it as "the reply was
    // not JSON" sends the reader looking for a fault in the model's writing
    // when the real fault is that it was given too little room to finish.
    if (truncated) {
      attempts.push({ attempt, problems: [TRUNCATED_PROBLEM], raw });
      throw new TruncatedReplyError(
        `The reply was cut off after ${options.maxOutputTokens} tokens, before it was finished.`,
        `Raise the room it has to answer in: "responseTokenBudget" for the description, or "benchmarkTokenBudget" for benchmark tables, in the config file.`,
      );
    }

    const parsed = parseJson(raw);
    if (parsed.problems.length > 0) {
      attempts.push({ attempt, problems: parsed.problems, raw });
      prompt = options.prompt + buildRetryNote(parsed.problems);
      continue;
    }

    const validated = schema.safeParse(parsed.value);
    if (!validated.success) {
      const problems = validated.error.issues.map(
        (issue) => `${issue.path.join(".") || "response"}: ${issue.message}`,
      );
      attempts.push({ attempt, problems, raw });
      prompt = options.prompt + buildRetryNote(problems);
      continue;
    }

    const extra = furtherChecks(validated.data);
    if (extra.length > 0) {
      attempts.push({ attempt, problems: extra, raw });
      prompt = options.prompt + buildRetryNote(extra);
      continue;
    }

    attempts.push({ attempt, problems: [], raw });
    return { value: validated.data, attempts };
  }

  const last = attempts[attempts.length - 1];
  throw new InvalidLlmOutputError(
    `The language model could not produce a valid answer after ${attempts.length} tries. Last problem: ${last?.problems.join("; ") ?? "unknown"}.`,
    `A larger or better-instructed model usually fixes this:\n  CATALOG_MODEL=<a larger model>\n\nRun \`catalog doctor\` to see what is installed.`,
  );
}

interface ParsedJson {
  value: unknown;
  problems: string[];
}

/**
 * Reads the reply as JSON. Models wrap JSON in prose or fences often enough
 * that one recovery attempt is worth it, but a reply that still will not parse
 * is rejected rather than repaired by guesswork.
 */
function parseJson(raw: string): ParsedJson {
  const trimmed = raw.trim();
  try {
    return { value: JSON.parse(trimmed), problems: [] };
  } catch {
    // Fall through to the fenced/embedded forms.
  }

  const fenced = /```(?:json)?\s*([\s\S]*?)```/.exec(trimmed);
  if (fenced?.[1] !== undefined) {
    try {
      return { value: JSON.parse(fenced[1]), problems: [] };
    } catch {
      // Fall through.
    }
  }

  const start = trimmed.indexOf("{");
  const end = trimmed.lastIndexOf("}");
  if (start !== -1 && end > start) {
    try {
      return { value: JSON.parse(trimmed.slice(start, end + 1)), problems: [] };
    } catch {
      // Fall through.
    }
  }

  return { value: null, problems: ["the reply was not JSON"] };
}

/**
 * Words that name what a model works on. A reply that talks about one the card
 * never mentions is describing something else.
 */
const MODALITY_CONCEPTS: Array<{ concept: string; stems: string[] }> = [
  { concept: "audio", stems: ["audio", "speech", "voice", "transcri", "acoustic", "asr", "spoken"] },
  { concept: "images", stems: ["image", "photo", "picture", "visual", "vision", "ocr"] },
  { concept: "video", stems: ["video", "frame", "clip", "footage"] },
  { concept: "music", stems: ["music", "song", "melody", "instrument"] },
  { concept: "alignment", stems: ["align"] },
  { concept: "embeddings", stems: ["embed", "vector", "retriev", "rerank"] },
  { concept: "speakers", stems: ["speaker", "diariz"] },
];

/**
 * Rules the schema cannot express on its own.
 *
 * `card` is the trimmed text the model was given. It is used to check that the
 * reply is about the model in front of it: a small model shown a worked example
 * will sometimes answer about the example instead, producing a fluent answer
 * for entirely the wrong model.
 */
export function proseChecks(value: Prose, card = ""): string[] {
  const problems: string[] = [];
  const categoryProblem = validateCategories(value.category as Category[]);
  if (categoryProblem !== null) problems.push(categoryProblem);

  // not_for exists to stop a wasted download. A restatement of the one-liner
  // does not do that.
  if (value.not_for.toLowerCase() === value.one_liner.toLowerCase()) {
    problems.push("not_for must name a misuse, not repeat one_liner");
  }

  // An echoed instruction reads as a real answer and passes every other rule,
  // so it is caught by name.
  for (const [field, text] of [
    ["one_liner", value.one_liner],
    ["not_for", value.not_for],
    ["deployment.reason", value.deployment.reason],
    ...value.use_for.map((entry, i) => [`use_for[${i}]`, entry] as const),
  ] as Array<readonly [string, string]>) {
    if (looksLikeInstruction(text)) {
      problems.push(
        `${field} repeats the instructions instead of describing the model. Write what this model actually does.`,
      );
    }
  }

  if (card !== "") {
    const haystack = card.toLowerCase();

    // Only one_liner is held to this. It states what the model *does*, so a
    // modality the README never mentions means it is describing something else.
    const summary = value.one_liner.toLowerCase();
    for (const { concept, stems } of MODALITY_CONCEPTS) {
      const claimed = stems.some((stem) => summary.includes(stem));
      // Any word from the same family counts: a card saying "transcript"
      // supports a reply saying "transcribing".
      const supported = stems.some((stem) => haystack.includes(stem));
      if (claimed && !supported) {
        problems.push(
          `one_liner talks about ${concept}, which this model's README never mentions. Describe the model in the README, not the example.`,
        );
        break;
      }
    }

    // not_for is the opposite case: it names what the model cannot do, so
    // pointing at a capability the README never mentions is exactly right.
    // "Not for generating images" is a fair warning on a text model. What is
    // not fair is the worked example's own sentence appearing on a card with
    // nothing to do with it, so only that is rejected.
    const warning = value.not_for.toLowerCase();
    for (const phrase of EXAMPLE_PHRASES) {
      if (!warning.includes(phrase.toLowerCase())) continue;
      // The example's wording is only wrong when the card has nothing to do
      // with what it is about. On a card that really does align audio, the same
      // sentence is the correct warning.
      const touched = MODALITY_CONCEPTS.filter(({ stems }) =>
        stems.some((stem) => phrase.toLowerCase().includes(stem)),
      );
      const supported = touched.some(({ stems }) => stems.some((stem) => haystack.includes(stem)));
      if (!supported) {
        problems.push(
          `not_for repeats the worked example word for word, and this README says nothing about it. Name a mistake someone would plausibly make with this model.`,
        );
        break;
      }
    }
  }

  return problems;
}

export async function extractProse(
  runtime: LlmRuntime,
  options: Omit<ExtractOptions, "system">,
  /** The trimmed card, so the reply can be checked for talking about something else. */
  card = "",
): Promise<ExtractionResult<Prose>> {
  return await extract(runtime, ProseSchema, { ...options, system: PROSE_SYSTEM_PROMPT }, (value) =>
    proseChecks(value, card),
  );
}

export async function extractBenchmarks(
  runtime: LlmRuntime,
  options: Omit<ExtractOptions, "system">,
): Promise<ExtractionResult<{ benchmarks: ExtractedBenchmark[] }>> {
  return await extract(runtime, BenchmarksSchema, {
    ...options,
    system: BENCHMARK_SYSTEM_PROMPT,
  });
}
