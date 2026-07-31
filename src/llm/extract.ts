import type { z } from "zod";
import { InvalidLlmOutputError } from "../errors.ts";
import type { GenerateRequest, LlmRuntime } from "./runtime.ts";
import { BENCHMARK_SYSTEM_PROMPT, buildRetryNote, PROSE_SYSTEM_PROMPT } from "./prompts.ts";
import {
  BenchmarksSchema,
  ProseSchema,
  validateCategories,
  type Category,
  type ExtractedBenchmark,
  type Prose,
} from "./schema.ts";

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

  for (let attempt = 1; attempt <= options.maxRetries + 1; attempt += 1) {
    const request: GenerateRequest = {
      system: options.system,
      prompt,
      contextTokens: options.contextTokens,
      maxOutputTokens: options.maxOutputTokens,
    };
    const raw = await runtime.generate(request);

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

/** Rules the schema cannot express on its own. */
export function proseChecks(value: Prose): string[] {
  const problems: string[] = [];
  const categoryProblem = validateCategories(value.category as Category[]);
  if (categoryProblem !== null) problems.push(categoryProblem);

  // not_for exists to stop a wasted download. A restatement of the one-liner
  // does not do that.
  if (value.not_for.toLowerCase() === value.one_liner.toLowerCase()) {
    problems.push("not_for must name a misuse, not repeat one_liner");
  }
  return problems;
}

export async function extractProse(
  runtime: LlmRuntime,
  options: Omit<ExtractOptions, "system">,
): Promise<ExtractionResult<Prose>> {
  return await extract(
    runtime,
    ProseSchema,
    { ...options, system: PROSE_SYSTEM_PROMPT },
    proseChecks,
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
