import { describe, expect, it } from "vitest";
import { InvalidLlmOutputError, RuntimeUnavailableError } from "../src/errors.ts";
import { extract, extractProse, proseChecks } from "../src/llm/extract.ts";
import type { GenerateRequest, LlmRuntime, RuntimeStatus } from "../src/llm/runtime.ts";
import { modelMaxContext, planContext } from "../src/llm/runtime.ts";
import {
  BenchmarksSchema,
  findBannedTerms,
  normaliseProse,
  ProseSchema,
  uncapitalise,
  validateCategories,
  type Prose,
} from "../src/llm/schema.ts";

/** A runtime that replies with a fixed script, so retry behaviour is testable. */
class ScriptedRuntime implements LlmRuntime {
  readonly name = "scripted";
  readonly model = "test-model";
  readonly requests: GenerateRequest[] = [];
  private readonly replies: string[];

  constructor(replies: string[]) {
    this.replies = [...replies];
  }

  async status(): Promise<RuntimeStatus> {
    return { reachable: true, modelPresent: true, modelMaxContext: 32768, version: "test", problem: null };
  }

  async requireReady(): Promise<RuntimeStatus> {
    return await this.status();
  }

  async generate(request: GenerateRequest): Promise<string> {
    this.requests.push(request);
    const reply = this.replies.shift();
    if (reply === undefined) throw new Error("the script ran out of replies");
    return reply;
  }
}

const VALID: Prose = {
  one_liner: "Takes an audio file and its transcript, returns the second each word is spoken.",
  use_for: ["subtitle timing", "karaoke", "marking up audio datasets"],
  not_for: "transcribing audio, you need a separate speech recognition model for that",
  category: ["audio-utility"],
  deployment: { where: "local", reason: "it is small and fast" },
  pairs_with: ["Qwen3-ASR"],
  supersedes_note: null,
};

const validJson = JSON.stringify(VALID);

const options = {
  prompt: "describe this model",
  contextTokens: 8192,
  maxOutputTokens: 700,
  maxRetries: 2,
};

describe("schema validation and retry", () => {
  it("accepts a well-formed reply on the first try", async () => {
    const runtime = new ScriptedRuntime([validJson]);
    const result = await extractProse(runtime, options);
    expect(result.value.one_liner).toBe(VALID.one_liner);
    expect(result.attempts).toHaveLength(1);
  });

  it("retries malformed JSON instead of rendering it", async () => {
    const runtime = new ScriptedRuntime(["this is not json at all", validJson]);
    const result = await extractProse(runtime, options);
    expect(result.attempts).toHaveLength(2);
    expect(result.attempts[0]?.problems).toContain("the reply was not JSON");
    expect(result.value.one_liner).toBe(VALID.one_liner);
  });

  it("retries a reply that is missing a required field", async () => {
    const missing = JSON.stringify({ ...VALID, not_for: undefined });
    const runtime = new ScriptedRuntime([missing, validJson]);
    const result = await extractProse(runtime, options);
    expect(result.attempts).toHaveLength(2);
    expect(result.attempts[0]?.problems.join(" ")).toContain("not_for");
  });

  it("retries a one-liner over the word limit", async () => {
    const tooLong = JSON.stringify({
      ...VALID,
      one_liner: Array.from({ length: 30 }, (_, i) => `word${i}`).join(" "),
    });
    const runtime = new ScriptedRuntime([tooLong, validJson]);
    const result = await extractProse(runtime, options);
    expect(result.attempts[0]?.problems.join(" ")).toContain("25 words or fewer");
    expect(result.value.one_liner).toBe(VALID.one_liner);
  });

  it("retries a one-liner full of jargon", async () => {
    const jargon = JSON.stringify({
      ...VALID,
      one_liner: "A non-autoregressive token classification model for forced alignment tasks.",
    });
    const runtime = new ScriptedRuntime([jargon, validJson]);
    const result = await extractProse(runtime, options);
    expect(result.attempts[0]?.problems.join(" ")).toContain("jargon");
  });

  it("retries a category outside the fixed vocabulary", async () => {
    const invented = JSON.stringify({ ...VALID, category: ["speech-alignment"] });
    const runtime = new ScriptedRuntime([invented, validJson]);
    const result = await extractProse(runtime, options);
    expect(result.attempts).toHaveLength(2);
    expect(result.value.category).toEqual(["audio-utility"]);
  });

  it("retries `base` combined with another category", async () => {
    const mixed = JSON.stringify({ ...VALID, category: ["base", "chat"] });
    const runtime = new ScriptedRuntime([mixed, validJson]);
    const result = await extractProse(runtime, options);
    expect(result.attempts[0]?.problems.join(" ")).toContain("base");
  });

  it("tells the model exactly what was wrong when it retries", async () => {
    const runtime = new ScriptedRuntime(["not json", validJson]);
    await extractProse(runtime, options);
    expect(runtime.requests[1]?.prompt).toContain("Your previous reply was rejected");
    expect(runtime.requests[1]?.prompt).toContain("the reply was not JSON");
  });

  it("fails rather than rendering anything after the retries run out", async () => {
    const runtime = new ScriptedRuntime(["nope", "still nope", "nope again"]);
    await expect(extractProse(runtime, options)).rejects.toBeInstanceOf(InvalidLlmOutputError);
  });

  it("makes exactly one more attempt than the retry budget", async () => {
    const runtime = new ScriptedRuntime(["a", "b", "c", "d"]);
    await expect(extractProse(runtime, { ...options, maxRetries: 3 })).rejects.toThrow();
    expect(runtime.requests).toHaveLength(4);
  });

  it("recovers JSON the model wrapped in a code fence", async () => {
    const runtime = new ScriptedRuntime(["```json\n" + validJson + "\n```"]);
    const result = await extractProse(runtime, options);
    expect(result.attempts).toHaveLength(1);
    expect(result.value.one_liner).toBe(VALID.one_liner);
  });

  it("recovers JSON the model surrounded with chatter", async () => {
    const runtime = new ScriptedRuntime([`Sure! Here you go:\n${validJson}\nHope that helps.`]);
    const result = await extractProse(runtime, options);
    expect(result.value.category).toEqual(["audio-utility"]);
  });

  it("accepts an empty benchmark list, which is a normal answer", async () => {
    const runtime = new ScriptedRuntime([JSON.stringify({ benchmarks: [] })]);
    const result = await extract(runtime, BenchmarksSchema, {
      ...options,
      system: "read tables",
    });
    expect(result.value.benchmarks).toEqual([]);
  });

  it("keeps benchmark scores as written rather than converting them", async () => {
    const runtime = new ScriptedRuntime([
      JSON.stringify({ benchmarks: [{ benchmark: "MMLU", score: "74.2", model: "Test-7B" }] }),
    ]);
    const result = await extract(runtime, BenchmarksSchema, { ...options, system: "read tables" });
    expect(result.value.benchmarks[0]?.score).toBe("74.2");
  });
});

describe("the plain-language rules", () => {
  it.each([
    "A non-autoregressive model for alignment.",
    "This is SOTA on every benchmark.",
    "A token classification model.",
    "Great zero-shot performance.",
    "Improves throughput on downstream tasks.",
  ])("catches jargon in %s", (text) => {
    expect(findBannedTerms(text).length).toBeGreaterThan(0);
  });

  it("does not catch an ordinary word that merely contains a banned one", () => {
    expect(findBannedTerms("A talented model that realignment never touched.")).toEqual([]);
  });

  it("passes the spec's own good example", () => {
    expect(findBannedTerms(VALID.one_liner)).toEqual([]);
    expect(ProseSchema.safeParse(VALID).success).toBe(true);
  });

  it("rejects the spec's own bad example", () => {
    const bad = {
      ...VALID,
      one_liner: "Non-autoregressive token classification model for forced alignment across 11 languages.",
    };
    expect(ProseSchema.safeParse(bad).success).toBe(false);
  });

  it("keeps `base` exclusive", () => {
    expect(validateCategories(["base"])).toBeNull();
    expect(validateCategories(["base", "chat"])).not.toBeNull();
  });

  it("rejects not_for that only repeats the one-liner", () => {
    const lazy = { ...VALID, not_for: VALID.one_liner };
    expect(proseChecks(lazy).join(" ")).toContain("not_for");
  });
});

describe("tidying the model's wording", () => {
  it("lowers an ordinary capitalised opener in a list item", () => {
    expect(uncapitalise("Batch timestamping recordings")).toBe("batch timestamping recordings");
  });

  it("leaves acronyms and model names alone", () => {
    expect(uncapitalise("MMLU scoring")).toBe("MMLU scoring");
    expect(uncapitalise("Qwen3-ASR pairing")).toBe("Qwen3-ASR pairing");
  });

  it("changes nothing but the wording", () => {
    const tidied = normaliseProse({ ...VALID, use_for: ["Subtitle timing", "Karaoke"] });
    expect(tidied.use_for).toEqual(["subtitle timing", "karaoke"]);
    expect(tidied.category).toEqual(VALID.category);
    expect(tidied.one_liner).toBe(VALID.one_liner);
  });
});

describe("planning the window", () => {
  it("reserves the reply before measuring the card", () => {
    const plan = planContext({
      requestedWindow: 16384,
      modelMaxContext: 262144,
      outputTokens: 700,
      overheadTokens: 700,
    });
    expect(plan.windowTokens).toBe(16384);
    expect(plan.cardBudgetTokens).toBe(16384 - 700 - 700);
    expect(plan.cappedByModel).toBe(false);
  });

  it("is capped by what the model itself supports", () => {
    const plan = planContext({
      requestedWindow: 16384,
      modelMaxContext: 4096,
      outputTokens: 700,
      overheadTokens: 700,
    });
    expect(plan.windowTokens).toBe(4096);
    expect(plan.cappedByModel).toBe(true);
  });

  it("never reports a negative budget", () => {
    const plan = planContext({
      requestedWindow: 512,
      modelMaxContext: 512,
      outputTokens: 700,
      overheadTokens: 700,
    });
    expect(plan.cardBudgetTokens).toBe(0);
  });

  it("reads the model's maximum from the namespaced key the runtime reports", () => {
    expect(modelMaxContext({ model_info: { "qwen3.context_length": 262144 } })).toBe(262144);
    expect(modelMaxContext({ model_info: { "llama.context_length": 8192 } })).toBe(8192);
    expect(modelMaxContext({ model_info: {} })).toBeNull();
    expect(modelMaxContext({})).toBeNull();
  });
});

describe("a missing runtime", () => {
  it("is fatal rather than degrading to numbers only", async () => {
    const dead: LlmRuntime = {
      name: "dead",
      model: "test-model",
      async status() {
        return {
          reachable: false,
          modelPresent: false,
          modelMaxContext: null,
          version: null,
          problem: "nothing is listening",
        };
      },
      async requireReady(): Promise<RuntimeStatus> {
        throw new RuntimeUnavailableError("nothing is listening", "ollama serve");
      },
      async generate() {
        throw new Error("should never be called");
      },
    };
    await expect(dead.requireReady()).rejects.toBeInstanceOf(RuntimeUnavailableError);
  });
});
