import { describe, expect, it } from "vitest";
import { nearestMatch, parseArgs } from "../src/cli/args.ts";
import { UsageError } from "../src/errors.ts";
import { parseModelRef } from "../src/hf/url.ts";
import { parseSizeFromName } from "../src/model/params.ts";

describe("reading a model reference", () => {
  it.each([
    "https://huggingface.co/Qwen/QwQ-32B",
    "https://huggingface.co/Qwen/QwQ-32B/",
    "http://huggingface.co/Qwen/QwQ-32B",
    "https://www.huggingface.co/Qwen/QwQ-32B",
    "https://hf.co/Qwen/QwQ-32B",
    "huggingface.co/Qwen/QwQ-32B",
    "Qwen/QwQ-32B",
    "https://huggingface.co/models/Qwen/QwQ-32B",
  ])("accepts %s", (input) => {
    expect(parseModelRef(input).repoId).toBe("Qwen/QwQ-32B");
  });

  it("keeps a revision named in the path", () => {
    const ref = parseModelRef("https://huggingface.co/Qwen/QwQ-32B/tree/refs%2Fpr%2F1");
    expect(ref.repoId).toBe("Qwen/QwQ-32B");
    expect(ref.revision).toBe("refs%2Fpr%2F1");
  });

  it("ignores the file part of a blob URL", () => {
    const ref = parseModelRef("https://huggingface.co/Qwen/QwQ-32B/blob/main/config.json");
    expect(ref.repoId).toBe("Qwen/QwQ-32B");
    expect(ref.revision).toBe("main");
  });

  it("accepts an explicit revision after an @", () => {
    const ref = parseModelRef("Qwen/QwQ-32B@976055f8");
    expect(ref.repoId).toBe("Qwen/QwQ-32B");
    expect(ref.revision).toBe("976055f8");
  });

  it("has no revision when none was given", () => {
    expect(parseModelRef("Qwen/QwQ-32B").revision).toBeNull();
  });

  it("says plainly that a dataset is not a model", () => {
    expect(() => parseModelRef("https://huggingface.co/datasets/squad")).toThrow(/dataset/);
  });

  it("says plainly that another site is not supported", () => {
    expect(() => parseModelRef("https://example.com/Qwen/QwQ-32B")).toThrow(/not Hugging Face/);
  });

  it.each(["", "QwQ-32B", "https://huggingface.co/Qwen"])("rejects %s with an example", (input) => {
    try {
      parseModelRef(input);
      expect.unreachable("should have thrown");
    } catch (error) {
      expect(error).toBeInstanceOf(UsageError);
      expect((error as UsageError).fix).toContain("catalog query");
    }
  });
});

describe("reading size out of a name", () => {
  it.each([
    ["Qwen3.6-35B-A3B", 35e9, 3e9],
    ["Qwen-AgentWorld-35B-A3B", 35e9, 3e9],
    ["QwQ-32B", 32e9, null],
    ["Qwen3-ForcedAligner-0.6B-hf", 0.6e9, null],
    ["Qwen3-Embedding-0.6B", 0.6e9, null],
    ["SmolLM-135M", 135e6, null],
    ["Llama-3.1-8B-Instruct", 8e9, null],
  ])("reads %s", (name, total, active) => {
    const hint = parseSizeFromName(name);
    expect(hint.total).toBe(total);
    expect(hint.active).toBe(active);
  });

  it("does not read a version number as a size", () => {
    // "Qwen2.5" must not be mistaken for 2.5 billion parameters.
    expect(parseSizeFromName("Qwen2.5-VL-7B-Instruct").total).toBe(7e9);
  });

  it("flags an 8x7B name as a mixture without inventing a total", () => {
    const hint = parseSizeFromName("Mixtral-8x7B-Instruct-v0.1");
    expect(hint.mixtureOfExperts).toBe(true);
    expect(hint.total).toBeNull();
  });

  it("reports nothing when the name carries no size", () => {
    const hint = parseSizeFromName("some-model-name");
    expect(hint.total).toBeNull();
    expect(hint.active).toBeNull();
  });
});

describe("command line flags", () => {
  it("reads the global flags", () => {
    const parsed = parseArgs(["fit", "Qwen/QwQ-32B", "--json", "--technical", "--no-color", "--refresh"]);
    expect(parsed.command).toBe("fit");
    expect(parsed.positionals).toEqual(["Qwen/QwQ-32B"]);
    expect(parsed.flags).toMatchObject({
      json: true,
      technical: true,
      color: false,
      refresh: true,
    });
  });

  it.each([
    [["--context", "32768"], 32768],
    [["--context=32768"], 32768],
    [["--context", "32_768"], 32768],
  ])("reads %s as a context length", (args, expected) => {
    expect(parseArgs(["fit", "x/y", ...args]).flags.context).toBe(expected);
  });

  it.each([["--context", "many"], ["--context", "-5"], ["--context", "1.5"]])(
    "rejects a context of %s with an example",
    (...args) => {
      try {
        parseArgs(["fit", "x/y", ...args]);
        expect.unreachable("should have thrown");
      } catch (error) {
        expect(error).toBeInstanceOf(UsageError);
        expect((error as UsageError).fix).toContain("--context");
      }
    },
  );

  it("suggests the nearest option for a typo", () => {
    expect(() => parseArgs(["fit", "x/y", "--jsn"])).toThrow(/Did you mean --json/);
  });

  it("suggests the nearest command for a typo", () => {
    expect(nearestMatch("spec", ["fit", "specs", "cache"])).toBe("specs");
    expect(nearestMatch("fitt", ["fit", "specs", "cache"])).toBe("fit");
  });

  it("suggests nothing when nothing is close", () => {
    expect(nearestMatch("wildlyunrelated", ["fit", "specs", "cache"])).toBeNull();
  });
});
