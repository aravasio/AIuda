import { describe, expect, it } from "vitest";
import { classify } from "../src/classify/index.ts";
import { normaliseModelInfo } from "../src/hf/client.ts";
import { parseArchitecture } from "../src/model/architecture.ts";
import { collectWeightVariants } from "../src/model/weights.ts";
import type { RepoSnapshot } from "../src/hf/types.ts";
import { loadRecordedError, loadSnapshot } from "./helpers/fixtures.ts";

describe("repo classification", () => {
  it("routes a plain model repo as its own subject", () => {
    const result = classify(loadSnapshot("standard-dense"));
    expect(result.kind).toBe("standard");
    expect(result.flags.hasConfig).toBe(true);
    expect(result.flags.hasWeights).toBe(true);
    expect(result.flags.multimodal).toBe(false);
    expect(result.unavailable).toEqual([]);
  });

  it("routes a third-party quantisation to its base model", () => {
    const snapshot = loadSnapshot("third-party-gguf");
    const result = classify(snapshot);
    expect(result.kind).toBe("quantization");
    expect(result.flags.ggufOnly).toBe(true);
    expect(result.flags.thirdParty).toBe(true);
    expect(result.baseModel).toBe("Qwen/Qwen2.5-7B-Instruct");
    // No config.json of its own: the fields it cannot supply are named.
    expect(result.flags.hasConfig).toBe(false);
    expect(result.unavailable).toContain("layer count");
  });

  it("marks a quantisation by the base model's own authors as first-party", () => {
    const snapshot = withInfo(loadSnapshot("third-party-gguf"), {
      repoId: "Qwen/Qwen2.5-7B-Instruct-GGUF",
      author: "Qwen",
    });
    const result = classify(snapshot);
    expect(result.kind).toBe("quantization");
    expect(result.flags.thirdParty).toBe(false);
  });

  it("finds the language-model fields nested under text_config", () => {
    const snapshot = loadSnapshot("moe-multimodal");
    const result = classify(snapshot);
    expect(result.kind).toBe("standard");
    expect(result.flags.multimodal).toBe(true);

    const architecture = parseArchitecture(snapshot.config);
    expect(architecture?.nested).toBe(true);
    expect(architecture?.layers?.source).toBe("config.json:text_config");
    expect(architecture?.layers?.value).toBe(40);
    expect(architecture?.kvHeads?.value).toBe(2);
    expect(architecture?.headDim?.value).toBe(256);
    expect(architecture?.moe?.experts).toBe(256);
    expect(architecture?.moe?.expertsPerToken).toBe(8);
  });

  it("keeps top-level fields when a multimodal repo states them there", () => {
    const snapshot = loadSnapshot("vision");
    const architecture = parseArchitecture(snapshot.config);
    expect(architecture?.nested).toBe(false);
    expect(architecture?.layers?.source).toBe("config.json");
    expect(classify(snapshot).flags.multimodal).toBe(false);
  });

  it("routes an adapter to its base model and refuses a standalone footprint", () => {
    const result = classify(loadSnapshot("adapter"));
    expect(result.kind).toBe("adapter");
    expect(result.unavailable.join(" ")).toContain("adapter is loaded on top of its base model");
  });

  it("degrades a repo with no config.json instead of inventing fields", () => {
    const snapshot = loadSnapshot("third-party-gguf");
    expect(snapshot.config).toBeNull();
    expect(parseArchitecture(snapshot.config)).toBeNull();
    const result = classify(snapshot);
    expect(result.unavailable).toEqual(
      expect.arrayContaining([
        "layer count",
        "key/value head count",
        "head dimension",
        "maximum context length",
      ]),
    );
  });

  it("reports a gated repo with the exact steps to unlock it", () => {
    const recorded = loadRecordedError("gated");
    expect(recorded.name).toBe("GatedRepoError");
    expect(recorded.error).toContain("gated");
    expect(recorded.error).not.toMatch(/401|403|undefined/);
  });

  it.each([
    ["manual", true],
    ["auto", true],
    [false, false],
    [null, false],
  ])("reads a gated value of %s from the API as %s", (raw, expected) => {
    // Hugging Face reports the gate as false or as the kind of gate it is,
    // so anything other than false means a token is needed.
    const info = normaliseModelInfo("owner/name", { sha: "abc123", gated: raw });
    expect(info.gated).toBe(expected);
  });

  it("notices an importance-matrix build", () => {
    const snapshot = loadSnapshot("third-party-gguf");
    const withImatrix: RepoSnapshot = {
      ...snapshot,
      files: [...snapshot.files, { path: "Model-IQ4_XS-imat.gguf", size: 100 }],
    };
    expect(classify(withImatrix).flags.imatrix).toBe(true);
  });
});

describe("weight variants", () => {
  it("keeps every GGUF quantisation separate", () => {
    const snapshot = loadSnapshot("third-party-gguf");
    const variants = collectWeightVariants(snapshot.files);
    const ggufFiles = snapshot.files.filter((f) => f.path.endsWith(".gguf"));

    // Each unsharded GGUF is a complete model on its own.
    expect(variants).toHaveLength(ggufFiles.length);
    const labels = variants.map((v) => v.label);
    expect(new Set(labels).size).toBe(labels.length);

    // Sizes must match the listing exactly, never a sum across quantisations.
    for (const variant of variants) {
      const expected = snapshot.files
        .filter((f) => variant.files.includes(f.path))
        .reduce((sum, f) => sum + (f.size ?? 0), 0);
      expect(variant.bytes).toBe(expected);
    }
  });

  it("does not merge Q4_0 with its Q4_0_4_8 neighbour", () => {
    const variants = collectWeightVariants([
      { path: "M-Q4_0.gguf", size: 4_440_000_000 },
      { path: "M-Q4_0_4_4.gguf", size: 4_430_000_000 },
      { path: "M-Q4_0_4_8.gguf", size: 4_431_000_000 },
    ]);
    expect(variants).toHaveLength(3);
    expect(variants.map((v) => v.bytes).sort()).toEqual([
      4_430_000_000, 4_431_000_000, 4_440_000_000,
    ]);
  });

  it("sums a sharded GGUF into one downloadable set", () => {
    const variants = collectWeightVariants([
      { path: "M-Q8_0-00001-of-00002.gguf", size: 5_000_000_000 },
      { path: "M-Q8_0-00002-of-00002.gguf", size: 3_000_000_000 },
    ]);
    expect(variants).toHaveLength(1);
    expect(variants[0]?.bytes).toBe(8_000_000_000);
    expect(variants[0]?.label).toBe("Q8_0");
  });

  it("sums the safetensors shards of a normal repo", () => {
    const snapshot = loadSnapshot("standard-dense");
    const variants = collectWeightVariants(snapshot.files, "bfloat16");
    expect(variants).toHaveLength(1);
    const expected = snapshot.files
      .filter((f) => f.path.endsWith(".safetensors"))
      .reduce((sum, f) => sum + (f.size ?? 0), 0);
    expect(variants[0]?.bytes).toBe(expected);
    expect(variants[0]?.quant).toBe("BF16");
  });

  it("ignores the legacy .bin copies when safetensors are present", () => {
    const variants = collectWeightVariants([
      { path: "model-00001-of-00002.safetensors", size: 1000 },
      { path: "model-00002-of-00002.safetensors", size: 1000 },
      { path: "pytorch_model-00001-of-00002.bin", size: 1000 },
      { path: "pytorch_model-00002-of-00002.bin", size: 1000 },
    ]);
    expect(variants).toHaveLength(1);
    expect(variants[0]?.bytes).toBe(2000);
    expect(variants[0]?.format).toBe("safetensors");
  });

  it("reports nothing when a repo carries no weights at all", () => {
    expect(collectWeightVariants([{ path: "README.md", size: 100 }])).toEqual([]);
  });
});

function withInfo(snapshot: RepoSnapshot, overrides: Record<string, unknown>): RepoSnapshot {
  return { ...snapshot, info: { ...snapshot.info, ...overrides } };
}
