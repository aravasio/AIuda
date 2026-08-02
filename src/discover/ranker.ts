import { RUNTIME_OVERHEAD_BYTES } from "../fit/math.ts";
import type { MachineSpecs } from "../specs/detect.ts";
import type { RegistryEntry } from "./types.ts";

/**
 * Rough estimate of total RAM needed.
 * Without the HF config.json (no layer count, kv heads, head dim),
 * we approximate the KV cache as a fraction of the weights.
 *
 * Dense models hold KV for every layer → more cache per token.
 * MoE models have fewer caching layers relative to total params.
 */
const KV_CACHE_FRACTION: Record<string, number> = {
  dense: 0.08,
  moe: 0.04,
};

const GB = 1_000_000_000;

export interface FitEstimate {
  totalBytes: number;
  fits: boolean;
  verdictKind: string;
}

export function estimateFit(
  entry: RegistryEntry,
  contextTokens: number,
  ramBytes: number,
): FitEstimate {
  const weightsBytes = entry.sizeGb * GB;
  const kvFraction = KV_CACHE_FRACTION[entry.architecture] ?? 0.06;
  const kvBytes = kvFraction * weightsBytes * (contextTokens / 8192);
  const totalBytes = weightsBytes + kvBytes + RUNTIME_OVERHEAD_BYTES;

  const THRESHOLDS = [
    { max: ramBytes * 0.8, kind: "runs-comfortably" },
    { max: ramBytes * 0.95, kind: "runs-tight" },
    { max: ramBytes, kind: "runs-slowly" },
  ];

  let fits = false;
  let verdictKind = "will-not-run";
  for (const t of THRESHOLDS) {
    if (totalBytes <= t.max) {
      fits = true;
      verdictKind = t.kind;
      break;
    }
  }

  return { totalBytes, fits, verdictKind };
}

/** Best benchmark available, or 0. */
export function bestBenchmark(entry: RegistryEntry): number {
  const b = entry.benchmarks;
  return b.sweBenchVerified ?? b.liveCodeBench ?? b.terminalBench ?? 0;
}

export interface RankedEntry extends RegistryEntry {
  fitBytes: number;
  fits: boolean;
  verdictKind: string;
  quality: number;
}

/** Rank entries: best benchmarked + fitting models first. */
export function rankEntries(
  entries: RegistryEntry[],
  specs: MachineSpecs,
  contextTokens: number,
): RankedEntry[] {
  const ram = specs.ramTotalBytes ?? 64 * GB; // fallback

  const scored = entries.map((e) => {
    const fit = estimateFit(e, contextTokens, ram);
    const quality = bestBenchmark(e);
    return { ...e, fitBytes: fit.totalBytes, fits: fit.fits, verdictKind: fit.verdictKind, quality };
  });

  scored.sort((a, b) => {
    // Fitting models first
    if (a.fits !== b.fits) return a.fits ? -1 : 1;
    // Then by quality descending
    if (b.quality !== a.quality) return b.quality - a.quality;
    // Then by size (smaller first when tied)
    return a.sizeGb - b.sizeGb;
  });

  return scored;
}

export function tierOf(sizeGb: number): "light" | "medium" | "heavy" {
  if (sizeGb < 20) return "light";
  if (sizeGb <= 35) return "medium";
  return "heavy";
}