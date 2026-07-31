import { describe, expect, it } from "vitest";
import type { Prose } from "../src/llm/schema.ts";
import { analyze, type Analysis } from "../src/pipeline/analyze.ts";
import { crossCheck } from "../src/pipeline/crosscheck.ts";
import { fakeMachine, loadSnapshot } from "./helpers/fixtures.ts";

const machine = fakeMachine();

/** QwQ-32B: 32.7 billion parameters, apache-2.0, dense. */
const dense: Analysis = analyze(loadSnapshot("standard-dense"), null, machine);
/** Qwen3.6-35B-A3B: 35.9 billion total, 3 billion active. */
const moe: Analysis = analyze(loadSnapshot("moe-multimodal"), null, machine);

function prose(overrides: Partial<Prose> = {}): Prose {
  return {
    one_liner: "Takes a question and returns an answer, with its reasoning shown first.",
    use_for: ["answering hard questions", "working through maths problems"],
    not_for: "quick replies, it thinks at length before answering",
    category: ["reasoning"],
    deployment: { where: "local", reason: "the weights are published" },
    pairs_with: null,
    supersedes_note: null,
    ...overrides,
  };
}

describe("structured data wins over the summary", () => {
  it("drops a parameter count that contradicts the repository", () => {
    const result = crossCheck(
      prose({ one_liner: "A 7B model that takes a question and returns a reasoned answer." }),
      dense,
    );
    expect(result.dropped).toHaveLength(1);
    expect(result.dropped[0]?.field).toBe("one_liner");
    expect(result.dropped[0]?.reason).toContain("32.7B");
  });

  it("does not average, blend or partially keep the dropped claim", () => {
    const result = crossCheck(prose({ one_liner: "A 7B reasoning model for hard questions." }), dense);
    // The repository's number is never rewritten into the model's sentence, and
    // the two are never combined into a third figure.
    expect(result.prose.one_liner).not.toContain("19");
    expect(result.prose.one_liner).not.toContain("32.7B");
    expect(result.dropped[0]?.claim).toContain("7B");
  });

  it("accepts a size the vendor rounds in the name", () => {
    // The repo holds 32.7 billion parameters and is called 32B.
    const result = crossCheck(prose({ one_liner: "A 32B model that answers hard questions." }), dense);
    expect(result.dropped).toHaveLength(0);
  });

  it("accepts the active count on a mixture-of-experts model", () => {
    // 35.9 billion total, 3 billion active. Naming either is fair.
    expect(crossCheck(prose({ one_liner: "A 36B model with 3B active per token." }), moe).dropped).toHaveLength(0);
  });

  it("drops a size that matches neither the total nor the active count", () => {
    const result = crossCheck(prose({ one_liner: "A 70B model for general use." }), moe);
    expect(result.dropped).toHaveLength(1);
  });

  it("drops a licence the repository contradicts", () => {
    const result = crossCheck(
      prose({ one_liner: "Answers hard questions, released under an MIT license." }),
      dense,
    );
    expect(result.dropped[0]?.reason).toContain("apache-2.0");
  });

  it("leaves a licence the repository agrees with", () => {
    const result = crossCheck(
      prose({ one_liner: "Answers questions, under the Apache license, with reasoning shown." }),
      dense,
    );
    expect(result.dropped).toHaveLength(0);
  });

  it("checks not_for as well as the one-liner", () => {
    const result = crossCheck(prose({ not_for: "small jobs, it is a 7B model" }), dense);
    expect(result.dropped.some((d) => d.field === "not_for")).toBe(true);
  });

  it("says nothing when the repository publishes no parameter count", () => {
    const noParams: Analysis = { ...dense, params: { ...dense.params, total: null, active: null } };
    expect(crossCheck(prose({ one_liner: "A 7B model for questions." }), noParams).dropped).toHaveLength(0);
  });
});

describe("structural facts win over the summary's judgement", () => {
  it("refuses `base` for a model the repository shows is instruction-tuned", () => {
    const instruct: Analysis = {
      ...dense,
      repoId: "Qwen/Qwen3-8B-Instruct",
      tags: ["conversational"],
    };
    const result = crossCheck(prose({ category: ["base"] }), instruct);
    expect(result.prose.category).not.toContain("base");
    expect(result.dropped.some((d) => d.field === "category")).toBe(true);
  });

  it("leaves `base` alone for a repository that really is a base model", () => {
    const base: Analysis = { ...dense, repoId: "Qwen/Qwen3-8B-Base", tags: [] };
    expect(crossCheck(prose({ category: ["base"] }), base).prose.category).toEqual(["base"]);
  });

  it("never leaves the category list empty after dropping one", () => {
    const instruct: Analysis = { ...dense, repoId: "Qwen/Qwen3-8B-Instruct", tags: ["conversational"] };
    const result = crossCheck(prose({ category: ["base"] }), instruct);
    expect(result.prose.category.length).toBeGreaterThan(0);
  });

  it("removes a model paired with itself", () => {
    const result = crossCheck(prose({ pairs_with: ["QwQ-32B", "Qwen3-ASR"] }), dense);
    expect(result.prose.pairs_with).toEqual(["Qwen3-ASR"]);
    expect(result.dropped.some((d) => d.field === "pairs_with")).toBe(true);
  });

  it("reports no pairing rather than an empty list", () => {
    expect(crossCheck(prose({ pairs_with: ["QwQ-32B"] }), dense).prose.pairs_with).toBeNull();
  });
});

describe("what the cross-check leaves alone", () => {
  it("passes clean prose through untouched", () => {
    const input = prose();
    const result = crossCheck(input, dense);
    expect(result.dropped).toEqual([]);
    expect(result.prose).toEqual(input);
  });

  it("does not modify the object it was given", () => {
    const input = prose({ category: ["base"], pairs_with: ["QwQ-32B"] });
    const before = JSON.stringify(input);
    crossCheck(input, { ...dense, repoId: "Qwen/Qwen3-8B-Instruct", tags: ["conversational"] });
    expect(JSON.stringify(input)).toBe(before);
  });
});

describe("bugs the eval set exposed", () => {
  it("drops a supersedes_note that only says the model has not been replaced", () => {
    // The real reply for QwQ-32B, which reads like information and carries none.
    const result = crossCheck(
      prose({ supersedes_note: "This model is part of the Qwen series and is not a newer version of itself" }),
      dense,
    );
    expect(result.prose.supersedes_note).toBeNull();
    expect(result.dropped.some((d) => d.field === "supersedes_note")).toBe(true);
  });

  it.each([
    "This is the current version of the model",
    "It has not been superseded",
    "QwQ-32B is the latest release in its family",
  ])("drops the vacuous note %s", (note) => {
    expect(crossCheck(prose({ supersedes_note: note }), dense).prose.supersedes_note).toBeNull();
  });

  it("keeps a note that names an actual replacement", () => {
    const note = "There is a newer version, Qwen/Qwen3-32B, which the README points to.";
    expect(crossCheck(prose({ supersedes_note: note }), dense).prose.supersedes_note).toBe(note);
  });

  it("keeps a note saying the model is mainly of research interest", () => {
    const note = "Mainly of research interest; for practical use prefer a newer instruction-tuned model.";
    expect(crossCheck(prose({ supersedes_note: note }), dense).prose.supersedes_note).toBe(note);
  });

  it("drops the base model listed as something to run alongside", () => {
    // QwQ-32B was built from Qwen2.5-32B. That is its ancestor, not its companion.
    const result = crossCheck(prose({ pairs_with: ["Qwen/Qwen2.5-32B"] }), dense);
    expect(result.prose.pairs_with).toBeNull();
    expect(result.dropped.some((d) => d.reason.includes("built from"))).toBe(true);
  });

  it("keeps a genuine companion model", () => {
    const result = crossCheck(prose({ pairs_with: ["Qwen/Qwen3-ASR"] }), dense);
    expect(result.prose.pairs_with).toEqual(["Qwen/Qwen3-ASR"]);
  });
});
