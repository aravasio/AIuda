import type { ModelInfo } from "../hf/types.ts";
import type { Architecture, FieldSource } from "./architecture.ts";

export interface ParamCount {
  /** Every parameter that must be held in memory. Drives the memory story. */
  total: number | null;
  /** Parameters actually used per token. On an MoE model this is far smaller and drives speed. */
  active: number | null;
  totalSource: FieldSource | null;
  activeSource: FieldSource | null;
}

/** What a name like "35B-A3B" or "0.6B" says about size. */
export interface NameSizeHint {
  total: number | null;
  active: number | null;
  /** True for "8x7B" style names, which announce a mixture without stating its real total. */
  mixtureOfExperts: boolean;
}

const B = 1_000_000_000;
const M = 1_000_000;

/**
 * Reads size out of the repo name. Vendors encode it consistently enough to rely
 * on as a fallback: "35B-A3B" is 35 billion total, 3 billion active per token.
 */
export function parseSizeFromName(name: string): NameSizeHint {
  // Split on separators only, never on the decimal point: "0.6B" is six
  // hundred million parameters, and splitting it would read it as six billion.
  // Version numbers like the "3.6" in "Qwen3.6" stay glued to their word and
  // therefore never match a size pattern.
  const segments = name.split(/[-_\s]+/);
  let total: number | null = null;
  let active: number | null = null;
  let mixtureOfExperts = false;

  for (const segment of segments) {
    const activeMatch = /^A(\d+(?:\.\d+)?)B$/i.exec(segment);
    if (activeMatch?.[1] !== undefined) {
      active = Number(activeMatch[1]) * B;
      continue;
    }
    // "8x7B" announces a mixture but the real total is not 8*7: the non-expert
    // layers are shared. Flag it and refuse to state a total from the name.
    if (/^\d+x\d+(?:\.\d+)?B$/i.test(segment)) {
      mixtureOfExperts = true;
      continue;
    }
    if (total === null) {
      const billions = /^(\d+(?:\.\d+)?)B$/i.exec(segment);
      if (billions?.[1] !== undefined) {
        total = Number(billions[1]) * B;
        continue;
      }
      const millions = /^(\d+(?:\.\d+)?)M$/i.exec(segment);
      if (millions?.[1] !== undefined) {
        total = Number(millions[1]) * M;
      }
    }
  }

  return { total, active, mixtureOfExperts };
}

/**
 * Resolves the parameter count from the most trustworthy source available.
 * safetensors metadata is an exact count published by the repo; the name is a
 * rounded marketing figure and is only used when nothing better exists.
 */
export function resolveParams(
  info: ModelInfo,
  architecture: Architecture | null,
  /** Repo name to read the size hint from. For a quantisation, pass the base model's name. */
  nameForHint: string,
): ParamCount {
  const hint = parseSizeFromName(nameForHint);

  let total: number | null = null;
  let totalSource: FieldSource | null = null;

  if (info.safetensorsTotalParams !== null && info.safetensorsTotalParams > 0) {
    total = info.safetensorsTotalParams;
    totalSource = "safetensors metadata";
  } else if (hint.total !== null) {
    total = hint.total;
    totalSource = "model name";
  }

  // Active params have no structured source: no config field states them and
  // deriving them from expert counts is unreliable across architectures.
  let active: number | null = null;
  let activeSource: FieldSource | null = null;
  const isMoe = architecture?.moe != null || hint.active !== null || hint.mixtureOfExperts;

  if (hint.active !== null) {
    active = hint.active;
    activeSource = "model name";
  } else if (!isMoe && total !== null) {
    // A dense model uses every parameter it holds.
    active = total;
    activeSource = "derived";
  }

  return { total, active, totalSource, activeSource };
}

/** True when memory and speed tell different stories and the output must say so. */
export function isMixtureOfExperts(architecture: Architecture | null, hint: NameSizeHint): boolean {
  if (architecture?.moe != null) return true;
  if (architecture?.modelType != null && /moe/i.test(architecture.modelType)) return true;
  return hint.active !== null || hint.mixtureOfExperts;
}
