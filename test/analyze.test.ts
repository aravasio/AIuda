import { describe, expect, it } from "vitest";
import { RUNTIME_OVERHEAD_BYTES } from "../src/fit/math.ts";
import { analyze, __testing } from "../src/pipeline/analyze.ts";
import { fakeMachine, loadSnapshot } from "./helpers/fixtures.ts";

const machine = fakeMachine();

describe("the deterministic pass, end to end", () => {
  it("reads a dense repo entirely from its own structured data", () => {
    const analysis = analyze(loadSnapshot("standard-dense"), null, machine);

    expect(analysis.repoId).toBe("Qwen/QwQ-32B");
    expect(analysis.license).toBe("apache-2.0");
    expect(analysis.sha).toHaveLength(40);
    expect(analysis.params.totalSource).toBe("safetensors metadata");
    expect(analysis.isMoe).toBe(false);

    const fit = analysis.fits[0]!;
    expect(fit.breakdown.weights?.source).toBe("repo file listing");
    // 64 layers, 8 KV heads, head dim 128, 8192 tokens, two bytes an element.
    expect(fit.breakdown.kv?.bytes).toBe(2 * 8192 * 64 * 8 * 128 * 2);
    expect(fit.breakdown.totalBytes).toBe(
      fit.breakdown.weights!.bytes + fit.breakdown.kv!.bytes + RUNTIME_OVERHEAD_BYTES,
    );
  });

  it("separates the memory story from the speed story on a mixture of experts", () => {
    const analysis = analyze(loadSnapshot("moe-multimodal"), null, machine);

    expect(analysis.isMoe).toBe(true);
    expect(analysis.params.total).toBeGreaterThan(30e9);
    expect(analysis.params.active).toBe(3e9);
    expect(analysis.notes.join(" ")).toMatch(/Memory follows the total, speed follows/);
  });

  it("charges only the caching layers on a hybrid-attention model", () => {
    const analysis = analyze(loadSnapshot("moe-multimodal"), null, machine);
    const kv = analysis.fits[0]!.breakdown.kv!;

    expect(kv.totalLayers).toBe(40);
    expect(kv.cachingLayers).toBe(10);
    expect(kv.layerSource).toBe("layer_types");
    expect(kv.bytes).toBe(2 * 8192 * 10 * 2 * 256 * 2);
    expect(analysis.notes.join(" ")).toMatch(/Only 10 of this model's 40 layers/);
  });

  it("reports the fixed layer state without folding it into the total unseen", () => {
    const analysis = analyze(loadSnapshot("world-model"), null, machine);
    const breakdown = analysis.fits[0]!.breakdown;
    const component = breakdown.sideComponents[0];

    expect(component).toBeDefined();
    expect(component!.bytes).toBeGreaterThan(0);
    expect(component!.counted).toBe(false);
    expect(component!.note).toContain("not added");
  });

  it("uses the base model's architecture for a quantisation with no config", () => {
    const analysis = analyze(
      loadSnapshot("third-party-gguf"),
      loadSnapshot("standard-dense"),
      machine,
    );

    expect(analysis.classification.kind).toBe("quantization");
    expect(analysis.architectureFrom).toBe("Qwen/QwQ-32B");
    expect(analysis.notes.join(" ")).toContain("Architecture read from the base model");
    // Once the base supplies them, those fields are no longer reported missing.
    expect(analysis.unavailable).not.toContain("layer count");
  });

  it("gives every quantisation in a catalogue its own real size", () => {
    const snapshot = loadSnapshot("third-party-gguf");
    const analysis = analyze(snapshot, null, machine);
    const byPath = new Map(snapshot.files.map((f) => [f.path, f.size ?? 0]));

    expect(analysis.fits.length).toBeGreaterThan(10);

    // Labels must be distinct. Sizes need not be: Q4_0_4_4 and Q4_0_8_8 are
    // the same quantisation laid out differently, so they weigh the same.
    const labels = analysis.fits.map((f) => f.variant?.label);
    expect(new Set(labels).size).toBe(labels.length);

    for (const fit of analysis.fits) {
      expect(fit.breakdown.weights?.source).toBe("repo file listing");
      // Each variant weighs exactly its own files, never a sum across
      // quantisations: a 7B model cannot weigh 18 GB at Q4.
      const own = fit.variant!.files.reduce((sum, path) => sum + (byPath.get(path) ?? 0), 0);
      expect(fit.breakdown.weights!.bytes).toBe(own);
      expect(fit.breakdown.weights!.bytes).toBeLessThan(16e9);
    }
  });

  it("never states a retained-accuracy percentage for a quantisation", () => {
    const analysis = analyze(loadSnapshot("third-party-gguf"), null, machine);
    for (const fit of analysis.fits) {
      expect(fit.qualityBand ?? "").not.toMatch(/\d+\s*%/);
    }
  });

  it("refuses a standalone total for an adapter", () => {
    const analysis = analyze(loadSnapshot("adapter"), null, machine);
    expect(analysis.classification.kind).toBe("adapter");
    expect(analysis.fits[0]?.verdict.kind).toBe("unknown");
    expect(analysis.notes.join(" ")).toContain("loaded on top of its base model");
  });

  it("scales the cache with the context the caller asked for", () => {
    const short = analyze(loadSnapshot("standard-dense"), null, machine, { contextTokens: 4096 });
    const long = analyze(loadSnapshot("standard-dense"), null, machine, { contextTokens: 32768 });
    expect(long.fits[0]!.breakdown.kv!.bytes).toBe(short.fits[0]!.breakdown.kv!.bytes * 8);
    // The weights do not move with context.
    expect(long.fits[0]!.breakdown.weights!.bytes).toBe(short.fits[0]!.breakdown.weights!.bytes);
  });

  it("warns when the requested context is longer than the model was trained for", () => {
    const analysis = analyze(loadSnapshot("standard-dense"), null, machine, {
      contextTokens: 131072,
    });
    expect(analysis.notes.join(" ")).toMatch(/trained for at most 40,960/);
  });

  it("is unaffected by which machine it runs on, except for the verdict", () => {
    const big = analyze(loadSnapshot("standard-dense"), null, fakeMachine({ ramTotalBytes: 512e9 }));
    const small = analyze(loadSnapshot("standard-dense"), null, fakeMachine({ ramTotalBytes: 8e9 }));
    expect(big.fits[0]!.breakdown.totalBytes).toBe(small.fits[0]!.breakdown.totalBytes);
    expect(big.fits[0]!.verdict.kind).not.toBe(small.fits[0]!.verdict.kind);
  });

  it("produces the same answer twice for the same input", () => {
    const first = analyze(loadSnapshot("moe-multimodal"), null, machine);
    const second = analyze(loadSnapshot("moe-multimodal"), null, machine);
    expect(JSON.stringify(first)).toBe(JSON.stringify(second));
  });
});

describe("surfacing a model that has been sitting still", () => {
  it("says how old a long-untouched repo is, from metadata rather than the summary", () => {
    // QwQ-32B: heavily downloaded, last touched March 2025.
    const analysis = analyze(loadSnapshot("standard-dense"), null, machine);
    expect(analysis.lastModified?.startsWith("2025-03")).toBe(true);
    expect(analysis.notes.join(" ")).toMatch(/Last updated .* months ago|over \d+ years ago/);
    expect(analysis.notes.join(" ")).toContain("check whether something newer");
  });

  it("counts the months between two dates", () => {
    expect(__testing.monthsSince("2025-03-11T00:00:00.000Z", new Date("2026-07-31T00:00:00.000Z"))).toBe(16);
    expect(__testing.monthsSince("2026-07-01T00:00:00.000Z", new Date("2026-07-31T00:00:00.000Z"))).toBe(0);
    expect(__testing.monthsSince(null)).toBeNull();
    expect(__testing.monthsSince("not a date")).toBeNull();
  });

  it("says nothing about the age of a freshly updated repo", () => {
    const fresh = loadSnapshot("standard-dense");
    const recent = {
      ...fresh,
      info: { ...fresh.info, lastModified: new Date().toISOString() },
    };
    expect(analyze(recent, null, machine).notes.join(" ")).not.toContain("Last updated");
  });
});
