import { describe, expect, it } from "vitest";
import {
  BYTES_PER_PARAM,
  bytesPerParam,
  qualityBand,
  quantFromFilename,
  quantFromTorchDtype,
} from "../src/model/quant.ts";
import {
  computeBreakdown,
  computeKvCache,
  computeWeightsBytes,
  estimateRecurrentStateBytes,
  NEGLIGIBLE_COMPONENT_FRACTION,
  RUNTIME_OVERHEAD_BYTES,
} from "../src/fit/math.ts";
import { classifyLayers, cachedTokensFor, normaliseLayerKind } from "../src/fit/layers.ts";
import type { Architecture } from "../src/model/architecture.ts";

/** Minimal architecture, so each test states only the fields it cares about. */
function architecture(overrides: Partial<Architecture> = {}): Architecture {
  return {
    modelType: "test",
    architectures: [],
    layers: { value: 32, source: "config.json" },
    hiddenSize: { value: 4096, source: "config.json" },
    attentionHeads: { value: 32, source: "config.json" },
    kvHeads: { value: 8, source: "config.json" },
    headDim: { value: 128, source: "config.json" },
    maxContext: { value: 32768, source: "config.json" },
    torchDtype: "bfloat16",
    moe: null,
    layerTypes: null,
    slidingWindow: null,
    useSlidingWindow: null,
    linearAttention: null,
    nested: false,
    missing: [],
    ...overrides,
  };
}

describe("weights size", () => {
  it("prefers a real file size over any estimate", () => {
    const result = computeWeightsBytes({
      realBytes: 15_231_233_536,
      paramsTotal: 7_615_616_512,
      quant: "BF16",
    });
    expect(result).toEqual({
      bytes: 15_231_233_536,
      source: "repo file listing",
      quant: "BF16",
    });
  });

  it("falls back to bytes-per-param and says so", () => {
    const result = computeWeightsBytes({ realBytes: null, paramsTotal: 8e9, quant: "Q4_K_M" });
    expect(result).toEqual({ bytes: 4_640_000_000, source: "estimate", quant: "Q4_K_M" });
  });

  it.each(Object.entries(BYTES_PER_PARAM))(
    "estimates %s at the documented rate",
    (format, perParam) => {
      const result = computeWeightsBytes({ realBytes: null, paramsTotal: 1e9, quant: format });
      expect(result?.bytes).toBe(Math.round(1e9 * perParam));
      expect(result?.source).toBe("estimate");
    },
  );

  it("returns nothing rather than guessing when the format is unknown", () => {
    expect(computeWeightsBytes({ realBytes: null, paramsTotal: 8e9, quant: "Q4_K_XL" })).toBeNull();
    expect(computeWeightsBytes({ realBytes: null, paramsTotal: null, quant: "BF16" })).toBeNull();
  });

  it("treats a zero-byte listing as no listing", () => {
    const result = computeWeightsBytes({ realBytes: 0, paramsTotal: 1e9, quant: "BF16" });
    expect(result?.source).toBe("estimate");
  });
});

describe("KV cache", () => {
  const uniform = (count: number) =>
    ({ classes: [{ kind: "full_attention" as const, count }], source: "num_hidden_layers" as const, totalLayers: count });

  it.each([
    // context, layers, kvHeads, headDim, expected bytes
    [1024, 32, 8, 128, 2 * 1024 * 32 * 8 * 128 * 2],
    [8192, 64, 8, 128, 2 * 8192 * 64 * 8 * 128 * 2],
    [32768, 40, 2, 256, 2 * 32768 * 40 * 2 * 256 * 2],
    [131072, 64, 8, 128, 2 * 131072 * 64 * 8 * 128 * 2],
  ])("is exact at %i tokens", (context, layers, kvHeads, headDim, expected) => {
    const result = computeKvCache({
      contextTokens: context,
      layers: uniform(layers),
      kvHeads,
      headDim,
    });
    expect(result.bytes).toBe(expected);
    expect(result.cachingLayers).toBe(layers);
  });

  it("scales linearly with context", () => {
    const at8k = computeKvCache({ contextTokens: 8192, layers: uniform(32), kvHeads: 8, headDim: 128 });
    const at16k = computeKvCache({ contextTokens: 16384, layers: uniform(32), kvHeads: 8, headDim: 128 });
    expect(at16k.bytes).toBe(at8k.bytes * 2);
  });

  it("honours a different KV element width", () => {
    const fp16 = computeKvCache({ contextTokens: 8192, layers: uniform(32), kvHeads: 8, headDim: 128 });
    const fp8 = computeKvCache({
      contextTokens: 8192,
      layers: uniform(32),
      kvHeads: 8,
      headDim: 128,
      bytesPerElement: 1,
    });
    expect(fp8.bytes).toBe(fp16.bytes / 2);
  });

  it("exceeds the weights at long context, which is the case that surprises people", () => {
    // A 7B model at Q4_K_M is about 4 GB of weights. Its cache at 128k is more.
    const weights = computeWeightsBytes({ realBytes: null, paramsTotal: 7e9, quant: "Q4_K_M" });
    const kv = computeKvCache({ contextTokens: 131072, layers: uniform(32), kvHeads: 32, headDim: 128 });
    expect(weights).not.toBeNull();
    expect(kv.bytes).toBeGreaterThan(weights!.bytes);
  });
});

describe("mixed attention kinds", () => {
  const hybrid = architecture({
    layers: { value: 40, source: "config.json" },
    kvHeads: { value: 2, source: "config.json" },
    headDim: { value: 256, source: "config.json" },
    layerTypes: Array.from({ length: 40 }, (_, i) =>
      (i + 1) % 4 === 0 ? "full_attention" : "linear_attention",
    ),
  });

  it("charges only the layers that keep a growing cache", () => {
    const layers = classifyLayers(hybrid);
    expect(layers).not.toBeNull();
    const result = computeKvCache({
      contextTokens: 8192,
      layers: layers!,
      kvHeads: 2,
      headDim: 256,
    });
    // 10 full-attention layers, not 40.
    expect(result.cachingLayers).toBe(10);
    expect(result.totalLayers).toBe(40);
    expect(result.bytes).toBe(2 * 8192 * 10 * 2 * 256 * 2);
  });

  it("is a quarter of what counting every layer would give", () => {
    const layers = classifyLayers(hybrid)!;
    const hybridBytes = computeKvCache({ contextTokens: 8192, layers, kvHeads: 2, headDim: 256 }).bytes;
    const naiveBytes = computeKvCache({
      contextTokens: 8192,
      layers: { classes: [{ kind: "full_attention", count: 40 }], source: "num_hidden_layers", totalLayers: 40 },
      kvHeads: 2,
      headDim: 256,
    }).bytes;
    expect(naiveBytes / hybridBytes).toBe(4);
  });

  it("caps sliding-window layers at the window, not the context", () => {
    const gemmaLike = architecture({
      layers: { value: 26, source: "config.json" },
      slidingWindow: 4096,
      layerTypes: Array.from({ length: 26 }, (_, i) =>
        (i + 1) % 6 === 0 ? "full_attention" : "sliding_attention",
      ),
    });
    const layers = classifyLayers(gemmaLike)!;
    const result = computeKvCache({
      contextTokens: 32768,
      layers,
      kvHeads: 4,
      headDim: 256,
      slidingWindow: 4096,
    });
    const sliding = result.components.find((c) => c.kind === "sliding_attention");
    const full = result.components.find((c) => c.kind === "full_attention");
    expect(sliding?.cachedTokens).toBe(4096);
    expect(full?.cachedTokens).toBe(32768);
  });

  it("ignores a window the config switches off", () => {
    const qwqLike = architecture({ slidingWindow: 32768, useSlidingWindow: false });
    const layers = classifyLayers(qwqLike)!;
    expect(layers.classes).toEqual([{ kind: "full_attention", count: 32 }]);
  });

  it("applies a window the config switches on", () => {
    const windowed = architecture({ slidingWindow: 4096, useSlidingWindow: true });
    const layers = classifyLayers(windowed)!;
    expect(layers.classes).toEqual([{ kind: "sliding_attention", count: 32 }]);
  });

  it("charges unrecognised layer kinds the full rate rather than nothing", () => {
    expect(normaliseLayerKind("something_new")).toBe("unknown");
    expect(cachedTokensFor("unknown", 8192, null)).toBe(8192);
  });

  it("reads vendor spellings of the same idea", () => {
    expect(normaliseLayerKind("linear_attention")).toBe("recurrent");
    expect(normaliseLayerKind("mamba")).toBe("recurrent");
    expect(normaliseLayerKind("sliding_attention")).toBe("sliding_attention");
    expect(normaliseLayerKind("full_attention")).toBe("full_attention");
  });
});

describe("fixed layer state", () => {
  const withRecurrent = architecture({
    linearAttention: {
      numKeyHeads: 16,
      keyHeadDim: 128,
      numValueHeads: 32,
      valueHeadDim: 128,
      convKernelDim: 4,
      stateDtype: "float32",
    },
  });

  it("is measured so its size can be judged", () => {
    const bytes = estimateRecurrentStateBytes(withRecurrent, 30);
    expect(bytes).not.toBeNull();
    expect(bytes!).toBeGreaterThan(0);
  });

  it("is reported but left out of the total while it stays small", () => {
    const breakdown = computeBreakdown({
      weights: { bytes: 70e9, source: "repo file listing", quant: "BF16" },
      architecture: architecture({
        ...withRecurrent,
        layerTypes: Array.from({ length: 40 }, (_, i) =>
          (i + 1) % 4 === 0 ? "full_attention" : "linear_attention",
        ),
      }),
      contextTokens: 8192,
    });
    const component = breakdown.sideComponents[0];
    expect(component).toBeDefined();
    expect(component!.counted).toBe(false);
    expect(component!.bytes).toBeGreaterThan(0);
    expect(breakdown.totalBytes).toBe(70e9 + breakdown.kv!.bytes + RUNTIME_OVERHEAD_BYTES);
  });

  it("is added to the total once it is a meaningful share of it", () => {
    const heavy = architecture({
      layers: { value: 40, source: "config.json" },
      layerTypes: Array.from({ length: 40 }, () => "linear_attention"),
      linearAttention: {
        numKeyHeads: 64,
        keyHeadDim: 1024,
        numValueHeads: 64,
        valueHeadDim: 1024,
        convKernelDim: 4,
        stateDtype: "float32",
      },
    });
    const breakdown = computeBreakdown({
      weights: { bytes: 2e9, source: "repo file listing", quant: "BF16" },
      architecture: heavy,
      contextTokens: 8192,
    });
    const component = breakdown.sideComponents[0];
    expect(component).toBeDefined();
    expect(component!.counted).toBe(true);
    const base = 2e9 + breakdown.kv!.bytes + RUNTIME_OVERHEAD_BYTES;
    expect(component!.bytes / base).toBeGreaterThanOrEqual(NEGLIGIBLE_COMPONENT_FRACTION);
    expect(breakdown.totalBytes).toBe(base + component!.bytes);
  });
});

describe("the total", () => {
  it("is weights plus cache plus the runtime allowance", () => {
    const breakdown = computeBreakdown({
      weights: { bytes: 4e9, source: "repo file listing", quant: "Q4_K_M" },
      architecture: architecture(),
      contextTokens: 8192,
    });
    expect(breakdown.totalBytes).toBe(4e9 + breakdown.kv!.bytes + RUNTIME_OVERHEAD_BYTES);
  });

  it("is withheld rather than half-computed when the cache is unknown", () => {
    const breakdown = computeBreakdown({
      weights: { bytes: 4e9, source: "repo file listing", quant: "Q4_K_M" },
      architecture: null,
      contextTokens: 8192,
    });
    expect(breakdown.totalBytes).toBeNull();
    expect(breakdown.unknown).toContain("KV cache size");
  });

  it("is withheld when the weights are unknown", () => {
    const breakdown = computeBreakdown({
      weights: null,
      architecture: architecture(),
      contextTokens: 8192,
    });
    expect(breakdown.totalBytes).toBeNull();
    expect(breakdown.unknown).toContain("weights size");
  });

  it("uses a configured runtime allowance in place of the default", () => {
    const breakdown = computeBreakdown({
      weights: { bytes: 4e9, source: "repo file listing", quant: "Q4_K_M" },
      architecture: architecture(),
      contextTokens: 8192,
      runtimeOverheadBytes: 2e9,
    });
    expect(breakdown.runtimeOverheadBytes).toBe(2e9);
    expect(breakdown.totalBytes).toBe(4e9 + breakdown.kv!.bytes + 2e9);
  });
});

describe("quantisation labels", () => {
  it.each([
    ["Qwen2.5-7B-Instruct-Q4_K_M.gguf", "Q4_K_M"],
    ["Qwen2.5-7B-Instruct-Q4_0_4_8.gguf", "Q4_0_4_8"],
    ["Qwen2.5-7B-Instruct-IQ4_XS.gguf", "IQ4_XS"],
    ["Qwen2.5-7B-Instruct-Q3_K_XL.gguf", "Q3_K_XL"],
    ["Qwen2.5-7B-Instruct-f16.gguf", "FP16"],
    ["Model-Q8_0-00001-of-00003.gguf", "Q8_0"],
  ])("reads %s as %s", (filename, expected) => {
    expect(quantFromFilename(filename)).toBe(expected);
  });

  it("reports nothing for a filename with no label", () => {
    expect(quantFromFilename("model.safetensors")).toBeNull();
  });

  it("maps torch dtypes onto table keys", () => {
    expect(quantFromTorchDtype("bfloat16")).toBe("BF16");
    expect(quantFromTorchDtype("float16")).toBe("FP16");
    expect(quantFromTorchDtype("float8_e4m3fn")).toBe("FP8");
    expect(quantFromTorchDtype(null)).toBeNull();
    expect(quantFromTorchDtype("something-else")).toBeNull();
  });

  it("bands quality qualitatively and never as a percentage", () => {
    expect(qualityBand("Q8_0")).toBe("effectively identical to the original");
    expect(qualityBand("Q6_K")).toBe("near-lossless");
    expect(qualityBand("Q3_K_M")).toBe("degrades meaningfully");
    // Every format the tool can size must also have a plain-language band,
    // and none of them may claim a retained-accuracy percentage.
    for (const format of Object.keys(BYTES_PER_PARAM)) {
      const band = qualityBand(format);
      expect(band, format).not.toBeNull();
      expect(band!).not.toMatch(/\d+\s*%/);
    }
  });

  it("bands labels the table does not list by name", () => {
    expect(qualityBand("Q4_K_L")).toBe("a noticeable quality loss, still usable");
    expect(qualityBand("IQ3_M")).toContain("packed tighter");
    expect(qualityBand("not-a-quant")).toBeNull();
  });

  it("declines to estimate bytes for a format it does not know", () => {
    expect(bytesPerParam("Q4_K_L")).toBeNull();
    expect(bytesPerParam("Q4_K_M")).toBe(0.58);
  });
});
