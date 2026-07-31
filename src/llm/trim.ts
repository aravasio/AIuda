import { CardTooLargeError } from "../errors.ts";

/**
 * Roughly four characters to a token for English prose with markdown around it.
 *
 * Note this is the *reading* model's tokenizer that matters, not the tokenizer
 * of the repo being described: the card is fed to whichever local model is
 * configured. A repo's own tokenizer.json would give a confidently precise
 * count of the wrong thing, so it is deliberately not used here. Precision is
 * not needed anyway — only the decision of what to drop.
 */
export const CHARS_PER_TOKEN = 4;

/** Counts tokens well enough to make a trim decision, erring toward overcounting. */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

/**
 * Lower keeps longer. Priority 0 never gets dropped: if the intro alone does
 * not fit, the run fails rather than describing a different model.
 */
export const Priority = {
  Essential: 0,
  Body: 20,
  Benchmarks: 50,
  Noise: 90,
} as const;

export interface Section {
  heading: string | null;
  level: number;
  body: string;
  priority: number;
  /** Position in the original card, so output order never changes. */
  index: number;
}

/** Which of the two extraction passes a trimmed card is being built for. */
export type Pass = "prose" | "benchmarks";

export interface TrimResult {
  text: string;
  tokens: number;
  budgetTokens: number;
  /** Headings that were removed, so the run can say what it did not read. */
  dropped: string[];
  /** True when code blocks were stripped, which is nearly always. */
  strippedCode: boolean;
}

const ESSENTIAL = /^(overview|introduction|intro|about|description|summary|model summary|highlights|key features|what'?s new|model card|tl;?dr)/i;
const NOISE =
  /^(quick ?start|getting started|installation|install|deployment|deploy|usage|how to use|run|running|inference|serving|citation|cite|license|licence|acknowledg|contact|contribut|disclaimer|ethical|limitations and bias|changelog|faq|api|examples?|best practices|requirements|dependencies)/i;
const BENCHMARKS = /^(benchmark|evaluation|eval|performance|result|comparison|score|metric|leaderboard)/i;

/**
 * Splits a model card into sections and scores each one.
 *
 * Markdown gives the structure for free, and the value in a card is not spread
 * evenly: the useful description sits at the top, the noise sits in the middle.
 */
export function parseSections(card: string): Section[] {
  const withoutFrontMatter = stripFrontMatter(card);
  const lines = withoutFrontMatter.split("\n");

  const sections: Section[] = [];
  let heading: string | null = null;
  let level = 0;
  let buffer: string[] = [];
  let inFence = false;

  const flush = (): void => {
    const body = buffer.join("\n").trim();
    if (heading === null && body === "") {
      buffer = [];
      return;
    }
    sections.push({
      heading,
      level,
      body,
      priority: priorityFor(heading, sections.length),
      index: sections.length,
    });
    buffer = [];
  };

  for (const line of lines) {
    // Headings inside a fenced block are code, not structure.
    if (/^\s*(```|~~~)/.test(line)) inFence = !inFence;

    const match = inFence ? null : /^(#{1,6})\s+(.*)$/.exec(line);
    if (match?.[1] !== undefined && match[2] !== undefined) {
      flush();
      level = match[1].length;
      heading = match[2].trim();
      continue;
    }
    buffer.push(line);
  }
  flush();

  return sections;
}

function priorityFor(heading: string | null, indexSoFar: number): number {
  // The text before any heading is the card's own opening line. It is the single
  // most useful part and is never dropped.
  if (heading === null) return Priority.Essential;
  const stripped = heading.replace(/[*_`#]/g, "").trim();
  if (ESSENTIAL.test(stripped)) return Priority.Essential;
  if (BENCHMARKS.test(stripped)) return Priority.Benchmarks;
  if (NOISE.test(stripped)) return Priority.Noise;
  // The first heading of a card is usually its title followed by the pitch.
  return indexSoFar <= 1 ? Priority.Essential : Priority.Body;
}

function stripFrontMatter(card: string): string {
  if (!card.startsWith("---")) return card;
  const end = card.indexOf("\n---", 3);
  if (end === -1) return card;
  return card.slice(end + 4);
}

/** Removes fenced code blocks, which are frequently most of a card by volume. */
export function stripCodeBlocks(text: string): string {
  return text
    .replace(/^\s*(```|~~~)[\s\S]*?^\s*\1[^\n]*$/gm, "")
    .replace(/^\s*(```|~~~)[\s\S]*$/m, "")
    .replace(/\n{3,}/g, "\n\n");
}

/** Removes markdown tables, which belong to the benchmark pass rather than the prose one. */
export function stripTables(text: string): string {
  return text.replace(/^\|.*\|[ \t]*$(\n^\|.*\|[ \t]*$)*/gm, "").replace(/\n{3,}/g, "\n\n");
}

/**
 * Keeps only sections holding a table that looks like results.
 *
 * Cards are full of tables that are not benchmarks: supported languages,
 * available checkpoints, feature matrices, download links. Handing those to the
 * model invites it to report "Cantonese" as a score, so they are filtered out
 * before it ever sees them rather than argued with afterwards.
 */
export function keepTables(sections: Section[]): Section[] {
  return sections.filter((section) => {
    const rows = section.body.split("\n").filter((line) => /^\s*\|.*\|/.test(line));
    if (rows.length < 3) return false;
    return numericCellShare(rows) >= MIN_NUMERIC_CELL_SHARE;
  });
}

/** A results table is mostly numbers. A feature matrix is mostly words. */
const MIN_NUMERIC_CELL_SHARE = 0.4;

function numericCellShare(rows: string[]): number {
  let numeric = 0;
  let total = 0;
  for (const row of rows) {
    // The separator row ("|---|---|") carries no data.
    if (/^\s*\|[\s:|-]*\|\s*$/.test(row)) continue;
    const cells = row.split("|").slice(1, -1);
    // The first cell is the row label, which is a name in every table.
    for (const cell of cells.slice(1)) {
      const value = cell.trim();
      if (value === "" || value === "-" || value === "—") continue;
      total += 1;
      if (/^[^A-Za-z]*\d[^A-Za-z]*$/.test(value)) numeric += 1;
    }
  }
  return total === 0 ? 0 : numeric / total;
}

/**
 * A benchmark score is a number, possibly with a percent sign, a range or an
 * error bar. Anything with letters in it is a feature, a language or a label
 * that the model mistook for a result.
 */
export function looksLikeScore(score: string): boolean {
  const value = score.trim();
  if (!/\d/.test(value)) return false;
  return /^[^A-Za-z]*$/.test(value);
}

export interface TrimOptions {
  budgetTokens: number;
  pass: Pass;
}

/**
 * Cuts a model card down to a budget, by priority and by rule.
 *
 * Trimming is always done before the model sees anything. Asking the model to
 * remove the noise is impossible, because reading the card in order to remove
 * the noise requires the noise to fit first.
 *
 * The order is fixed: code blocks, then the sections that are pure instructions,
 * then measure, then drop whole sections lowest-priority-first. Halving the card
 * blindly would be simpler and would sometimes discard exactly the paragraph
 * that says what the model does.
 */
export function trimCard(card: string, options: TrimOptions): TrimResult {
  const { budgetTokens, pass } = options;
  const dropped: string[] = [];

  let sections = parseSections(card);

  // Step 1: code blocks go first, whatever else happens.
  sections = sections.map((section) => ({ ...section, body: stripCodeBlocks(section.body) }));

  // Each pass receives only the sections it needs, which lowers the pressure on
  // both and improves both answers.
  if (pass === "benchmarks") {
    sections = keepTables(sections);
  } else {
    sections = sections.map((section) => ({ ...section, body: stripTables(section.body) }));
  }

  sections = sections.filter((section) => section.body.trim() !== "" || section.heading !== null);

  const render = (kept: Section[]): string =>
    kept
      .slice()
      .sort((a, b) => a.index - b.index)
      .map((section) =>
        section.heading === null ? section.body : `${"#".repeat(section.level)} ${section.heading}\n\n${section.body}`,
      )
      .join("\n\n")
      .replace(/\n{3,}/g, "\n\n")
      .trim();

  if (sections.length === 0) {
    throw new CardTooLargeError(
      pass === "benchmarks"
        ? "This card has no benchmark tables to read."
        : "This card has no readable text once code blocks are removed.",
    );
  }

  // Step 2: drop the sections that are pure instructions.
  let kept = sections;
  if (estimateTokens(render(kept)) > budgetTokens) {
    const withoutNoise = kept.filter((section) => section.priority < Priority.Noise);
    if (withoutNoise.length > 0) {
      for (const section of kept) {
        if (section.priority >= Priority.Noise && section.heading !== null) {
          dropped.push(section.heading);
        }
      }
      kept = withoutNoise;
    }
  }

  // Step 3 and 4: measure, then drop whole sections lowest-priority-first.
  // Dropping a section entirely is safer than cutting one in half, which
  // produces text that reads as complete while missing its conclusion.
  while (estimateTokens(render(kept)) > budgetTokens && kept.length > 1) {
    const droppable = kept.filter((section) => section.priority > Priority.Essential);
    if (droppable.length === 0) break;
    const worst = droppable.reduce((a, b) =>
      b.priority > a.priority || (b.priority === a.priority && b.index > a.index) ? b : a,
    );
    if (worst.heading !== null) dropped.push(worst.heading);
    kept = kept.filter((section) => section !== worst);
  }

  const text = render(kept);
  const tokens = estimateTokens(text);

  // A half-read card produces a summary that reads perfectly and describes a
  // different model. That is worse than no answer, because it is confidently
  // wrong and the user cannot tell.
  if (tokens > budgetTokens) {
    throw new CardTooLargeError(
      `This model card does not fit in the space the language model has, even after removing everything but its introduction. It needs about ${tokens.toLocaleString("en-US")} tokens and there is room for ${budgetTokens.toLocaleString("en-US")}. No summary was produced, because a summary of half a card describes a different model.`,
      `Give the model a larger window by raising "llmContextTokens" in the config file, or set a model with more room:\n  CATALOG_MODEL=<a model with a longer context>`,
    );
  }

  if (text.trim() === "") {
    throw new CardTooLargeError("Nothing readable was left of this card after trimming.");
  }

  return { text, tokens, budgetTokens, dropped, strippedCode: true };
}
