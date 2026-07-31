import { z } from "zod";

/**
 * Fixed category list. Adding one requires updating the glossary in the same
 * commit, so the vocabulary and its explanations never drift apart.
 */
export const CATEGORIES = [
  "chat",
  "reasoning",
  "coding",
  "agentic",
  "vision",
  "video",
  "speech-to-text",
  "text-to-speech",
  "audio-utility",
  "embedding",
  "reranker",
  "image-generation",
  "world-model",
  "guard",
  "base",
] as const;

export type Category = (typeof CATEGORIES)[number];

/**
 * Words that mean nothing to someone who has not already read the field's
 * literature. Banned from the one-liner unless the sentence defines them on the
 * spot, which is what the `unless defined inline` rule amounts to in practice.
 */
export const BANNED_TERMS = [
  "SOTA",
  "autoregressive",
  "non-autoregressive",
  "token classification",
  "causal LM",
  "chain-of-thought",
  "distillation",
  "alignment",
  "RLHF",
  "post-training",
  "zero-shot",
  "OOD",
  "latent",
  "throughput",
  "downstream",
] as const;

export const MAX_ONE_LINER_WORDS = 25;

export function countWords(text: string): number {
  return text.trim().split(/\s+/).filter((w) => w.length > 0).length;
}

/** Returns the banned terms present in the text, so the retry can name them. */
export function findBannedTerms(text: string): string[] {
  const found: string[] = [];
  for (const term of BANNED_TERMS) {
    // Word boundaries, so "alignment" is caught but "realignment" is not, and
    // "latent" is caught but "talented" is not.
    const pattern = new RegExp(`(^|[^A-Za-z-])${escapeRegex(term)}([^A-Za-z-]|$)`, "i");
    if (pattern.test(text)) found.push(term);
  }
  return found;
}

function escapeRegex(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const oneLiner = z
  .string()
  .trim()
  .min(10, "one_liner is too short to say anything")
  .refine((value) => countWords(value) <= MAX_ONE_LINER_WORDS, {
    message: `one_liner must be ${MAX_ONE_LINER_WORDS} words or fewer`,
  })
  .refine((value) => findBannedTerms(value).length === 0, {
    message: "one_liner uses jargon that means nothing to a newcomer",
  });

/**
 * The prose the language model is allowed to produce. Every field here is
 * words. Not one of them is a number, a size, a licence or a context length:
 * those come from config.json and repo metadata, and the model is never asked
 * for them.
 */
export const ProseSchema = z.object({
  one_liner: oneLiner,
  use_for: z.array(z.string().trim().min(3)).min(2).max(4),
  not_for: z.string().trim().min(5),
  category: z.array(z.enum(CATEGORIES)).min(1),
  deployment: z.object({
    where: z.enum(["local", "hosted", "either"]),
    reason: z.string().trim().min(5),
  }),
  pairs_with: z.array(z.string().trim().min(1)).nullable(),
  supersedes_note: z.string().trim().min(5).nullable(),
});

export type Prose = z.infer<typeof ProseSchema>;

/**
 * `base` means the model has had no instruction tuning: it completes text and
 * will not hold a conversation. That is a different enough thing that mixing it
 * with another category would hide it.
 */
export const BaseModelWarning =
  "This is a base model. It completes text rather than answering questions, and needs fine-tuning before it is useful as an assistant.";

export function validateCategories(categories: Category[]): string | null {
  if (categories.includes("base") && categories.length > 1) {
    return "category may not combine `base` with anything else: a base model is not a chat model";
  }
  return null;
}

/** Where a benchmark number came from, which decides how much it is worth. */
export const BenchmarkSourceSchema = z.enum(["model-index", "card-table", "local"]);
export type BenchmarkSource = z.infer<typeof BenchmarkSourceSchema>;

export const BenchmarkSchema = z.object({
  benchmark: z.string().trim().min(1),
  /** Kept as a string: some scores are percentages, some are "62.3 / 71.0". */
  score: z.string().trim().min(1),
  /** The model the row refers to, since vendor tables compare against rivals. */
  model: z.string().trim().min(1).nullable(),
});

export const BenchmarksSchema = z.object({
  benchmarks: z.array(BenchmarkSchema),
});

export type ExtractedBenchmark = z.infer<typeof BenchmarkSchema>;

/**
 * Fragments that only appear in instructions about the answer, never in an
 * answer. A model given a very long card will sometimes return the rule for a
 * field instead of filling it in, and the result passes every other check: a
 * sentence describing what a sentence should contain is a well-formed sentence
 * of the right length with no jargon in it.
 */
const INSTRUCTION_FRAGMENTS = [
  "what goes in",
  "what comes out",
  "words maximum",
  "at most 25 words",
  "one sentence",
  "two to four",
  "short phrases",
  "or null",
  "one or more of",
  "the most likely mistake",
  "what to use instead",
  "concrete things",
  "fill in",
  "filled in",
  "your answer",
  "the readme above",
  "the model above",
] as const;

/** True when the text reads as a rule about the field rather than its value. */
export function looksLikeInstruction(text: string): boolean {
  const value = text.toLowerCase();
  if (value.includes("…") || value.includes("...")) return true;
  return INSTRUCTION_FRAGMENTS.some((fragment) => value.includes(fragment));
}

/**
 * Language models capitalise list items like sentence openers. These are read
 * as a comma-separated run ("subtitle timing, karaoke, marking up datasets"),
 * where mid-list capitals look like a mistake.
 *
 * Only an ordinary capitalised word is lowered. Acronyms and model names, which
 * do not match the shape, are left exactly as written.
 */
export function uncapitalise(text: string): string {
  const [first, ...rest] = text.split(" ");
  if (first === undefined) return text;
  if (!/^[A-Z][a-z]+$/.test(first)) return text;
  return [first.toLowerCase(), ...rest].join(" ");
}

export function normaliseProse(prose: Prose): Prose {
  return {
    ...prose,
    use_for: prose.use_for.map(uncapitalise),
    not_for: uncapitalise(prose.not_for),
    deployment: { ...prose.deployment, reason: uncapitalise(prose.deployment.reason) },
  };
}

/** One benchmark result with its provenance attached, ready to render. */
export interface LabelledBenchmark {
  benchmark: string;
  score: string;
  model: string | null;
  source: BenchmarkSource;
}
