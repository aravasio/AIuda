import { fastMemoryBytes, type MachineSpecs } from "../specs/detect.ts";
import { COMFORTABLE_HEADROOM, type Breakdown } from "./math.ts";

export type VerdictKind =
  | "runs-comfortably"
  | "runs-tight"
  | "runs-slowly"
  | "split-possible"
  | "will-not-run"
  | "unknown";

export interface Verdict {
  kind: VerdictKind;
  /** One line in plain English, safe to print on its own. */
  summary: string;
  /** Why the verdict came out this way, in terms of the numbers involved. */
  detail: string;
  /** Fast memory the model was measured against, when there is any. */
  fastMemoryBytes: number | null;
  systemMemoryBytes: number | null;
}

export function decideVerdict(
  breakdown: Breakdown,
  specs: MachineSpecs,
  options: { isMoe: boolean },
): Verdict {
  const fast = fastMemoryBytes(specs);
  const ram = specs.ramTotalBytes;
  const total = breakdown.totalBytes;

  if (total === null) {
    return {
      kind: "unknown",
      summary: "Cannot say. Some of the numbers this needs are not published by the repo.",
      detail: `Missing: ${breakdown.unknown.join(", ")}.`,

      fastMemoryBytes: fast,
      systemMemoryBytes: ram,
    };
  }

  if (fast === null && ram === null) {
    return {
      kind: "unknown",
      summary: "Cannot say. This machine's memory could not be read.",
      detail: "Write a machine override file to supply it. Run `catalog specs` to see the template.",
      fastMemoryBytes: null,
      systemMemoryBytes: null,
    };
  }

  const unified = specs.memoryKind === "unified";

  if (fast !== null && total <= fast * (1 - COMFORTABLE_HEADROOM)) {
    return {
      kind: "runs-comfortably",
      summary: "Runs comfortably.",
      detail: `${gb(total)} needed against ${gb(fast)} of ${unified ? "unified memory usable by the GPU" : "VRAM"}, leaving room for longer context.`,
      fastMemoryBytes: fast,
      systemMemoryBytes: ram,
    };
  }

  if (fast !== null && total <= fast) {
    return {
      kind: "runs-tight",
      summary: "Runs, tight.",
      detail: `${gb(total)} needed against ${gb(fast)} of ${unified ? "unified memory usable by the GPU" : "VRAM"}. Expect trouble past ${breakdown.contextTokens.toLocaleString("en-US")} tokens of context.`,
      fastMemoryBytes: fast,
      systemMemoryBytes: ram,
    };
  }

  const fitsRam = ram !== null && total <= ram;

  // A mixture-of-experts model is the one case where not fitting the GPU is not
  // the end of it: only a few experts are touched per token, so the idle ones
  // can sit in system RAM at a far smaller speed cost than a dense model pays.
  if (options.isMoe && fitsRam && fast !== null) {
    return {
      kind: "split-possible",
      summary: "Split possible.",
      detail: `${gb(total)} is more than the ${gb(fast)} of ${unified ? "unified memory usable by the GPU" : "VRAM"}, but it fits in ${gb(ram)} of system memory. Because this is a mixture-of-experts model, the experts can stay in RAM while the shared layers run on the GPU: llama.cpp's --n-cpu-moe or KTransformers do this.`,
      fastMemoryBytes: fast,
      systemMemoryBytes: ram,
    };
  }

  if (fitsRam) {
    return {
      kind: "runs-slowly",
      summary: "Runs slowly.",
      detail:
        fast === null
          ? `${gb(total)} fits in ${gb(ram)} of system memory, but there is no GPU here, so it runs on the CPU.`
          : `${gb(total)} fits in ${gb(ram)} of system memory but not in the ${gb(fast)} of VRAM, so most of it runs on the CPU.`,
      fastMemoryBytes: fast,
      systemMemoryBytes: ram,
    };
  }

  const ceiling = ram ?? fast ?? 0;
  return {
    kind: "will-not-run",
    summary: "Will not run.",
    detail: `${gb(total)} needed, ${gb(ceiling)} available. A smaller quantisation or a shorter context would lower the requirement.`,
    fastMemoryBytes: fast,
    systemMemoryBytes: ram,
  };
}

/**
 * Decimal GB, the unit Hugging Face reports file sizes in and the one a user
 * comparing against a download size will expect. Mixing the two conventions is
 * a 7% error, which is small enough to pass unnoticed and large enough to
 * change a verdict near the edge.
 */
export function gb(bytes: number): string {
  const value = bytes / 1e9;
  if (value < 10) return `${value.toFixed(1)} GB`;
  return `${Math.round(value)} GB`;
}

export function mb(bytes: number): string {
  return `${Math.round(bytes / 1e6)} MB`;
}

/** Picks GB or MB so small models are not all reported as "0.1 GB". */
export function humanBytes(bytes: number): string {
  return bytes < 1e9 ? mb(bytes) : gb(bytes);
}
