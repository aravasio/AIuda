import type { Prose } from "../llm/schema.ts";
import { formatParams } from "../render/format.ts";
import type { Analysis } from "./analyze.ts";

/** One claim the language model made that the repo's own data contradicts. */
export interface DroppedClaim {
  field: string;
  claim: string;
  reason: string;
}

export interface CrossCheckResult {
  prose: Prose;
  /** Claims removed because structured data disagreed. Never merged, never averaged. */
  dropped: DroppedClaim[];
}

/**
 * Checks the language model's prose against the repo's structured data.
 *
 * On any conflict the structured value wins and the model's version is dropped
 * outright. It is not blended, averaged or preferred-but-noted: a wrong number
 * that survives in any form is a wrong number the user will read.
 *
 * The model is instructed not to write numbers at all. This is the net that
 * catches it when it does anyway.
 */
export function crossCheck(prose: Prose, analysis: Analysis): CrossCheckResult {
  const dropped: DroppedClaim[] = [];
  const checked: Prose = {
    ...prose,
    use_for: [...prose.use_for],
    category: [...prose.category],
    pairs_with: prose.pairs_with === null ? null : [...prose.pairs_with],
  };

  // A parameter count in the prose is either right, in which case it is
  // redundant with the Size line, or wrong, in which case it is dangerous.
  for (const field of ["one_liner", "not_for"] as const) {
    const conflict = numericConflict(checked[field], analysis);
    if (conflict !== null) {
      dropped.push({ field, claim: checked[field], reason: conflict });
    }
  }

  const licenceConflict = statedLicenceConflict(checked.one_liner, analysis);
  if (licenceConflict !== null) {
    dropped.push({ field: "one_liner", claim: checked.one_liner, reason: licenceConflict });
  }

  // `base` is a structural fact, not an opinion: a repo whose metadata and name
  // both say it is instruction-tuned is not a base model, whatever the card
  // reads like.
  if (checked.category.includes("base") && looksInstructionTuned(analysis)) {
    dropped.push({
      field: "category",
      claim: "base",
      reason: `${analysis.repoId} is tagged and named as an instruction-tuned model, so it is not a base model.`,
    });
    checked.category = checked.category.filter((c) => c !== "base");
    if (checked.category.length === 0) checked.category = ["chat"];
  }

  // A model cannot pair with itself.
  if (checked.pairs_with !== null) {
    const own = analysis.repoId.split("/")[1]?.toLowerCase() ?? "";
    const filtered = checked.pairs_with.filter((name) => name.toLowerCase() !== own);
    if (filtered.length !== checked.pairs_with.length) {
      dropped.push({
        field: "pairs_with",
        claim: analysis.repoId,
        reason: "a model cannot be paired with itself",
      });
    }
    checked.pairs_with = filtered.length === 0 ? null : filtered;
  }

  return { prose: checked, dropped };
}

/**
 * Looks for a parameter count in the prose that disagrees with the repo's own.
 * Sizes are the number models get wrong most often and the one a reader is most
 * likely to act on.
 */
function numericConflict(text: string, analysis: Analysis): string | null {
  if (analysis.params.total === null) return null;

  const matches = text.matchAll(/\b(\d+(?:\.\d+)?)\s*([BM])\b(?!\w)/gi);
  for (const match of matches) {
    const value = Number(match[1]);
    const unit = match[2]?.toUpperCase();
    if (!Number.isFinite(value) || unit === undefined) continue;
    const claimed = unit === "B" ? value * 1e9 : value * 1e6;

    // Either the total or the active count is a fair thing to name.
    const acceptable = [analysis.params.total, analysis.params.active].filter(
      (v): v is number => v !== null,
    );
    const matchesSomething = acceptable.some((actual) => withinRounding(claimed, actual));
    if (!matchesSomething) {
      return `the card summary said ${match[0]}, but the repository reports ${formatParams(analysis.params.total)}${analysis.params.active !== null && analysis.params.active !== analysis.params.total ? ` with ${formatParams(analysis.params.active)} active` : ""}`;
    }
  }
  return null;
}

/** Vendors round in names, so "35B" for 35.95 billion is right, not a conflict. */
function withinRounding(claimed: number, actual: number): boolean {
  return Math.abs(claimed - actual) / actual <= 0.1;
}

function statedLicenceConflict(text: string, analysis: Analysis): string | null {
  if (analysis.license === null) return null;
  const licence = analysis.license.toLowerCase();
  const known = ["apache", "mit", "gpl", "bsd", "llama", "gemma", "cc-by", "proprietary"];
  for (const candidate of known) {
    const mentioned = new RegExp(`\\b${candidate}[\\w.-]*\\s+licen[cs]e`, "i").test(text);
    if (mentioned && !licence.includes(candidate)) {
      return `the card summary named a ${candidate} licence, but the repository declares ${analysis.license}`;
    }
  }
  return null;
}

function looksInstructionTuned(analysis: Analysis): boolean {
  const name = analysis.repoId.toLowerCase();
  if (/-(base|pt)$/.test(name)) return false;
  if (/instruct|chat|-it\b|thinking/.test(name)) return true;
  return analysis.tags.some((tag) => tag === "conversational");
}
