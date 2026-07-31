import { describe, expect, it } from "vitest";
import { InvalidLlmOutputError, RuntimeUnavailableError, TruncatedReplyError } from "../src/errors.ts";
import { extract, extractProse, proseChecks } from "../src/llm/extract.ts";
import type { GeneratedReply, GenerateRequest, LlmRuntime, RuntimeStatus } from "../src/llm/runtime.ts";
import { modelMaxContext, planContext } from "../src/llm/runtime.ts";
import {
  BenchmarksSchema,
  findBannedTerms,
  looksLikeInstruction,
  normaliseProse,
  ProseSchema,
  uncapitalise,
  validateCategories,
  type Prose,
} from "../src/llm/schema.ts";

/** Stands in for a reply the runtime cut off at the output limit. */
const TRUNCATED_MARKER = "<<TRUNCATED>>";

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

  async generate(request: GenerateRequest): Promise<GeneratedReply> {
    this.requests.push(request);
    const reply = this.replies.shift();
    if (reply === undefined) throw new Error("the script ran out of replies");
    // A scripted reply is complete unless the test marks it cut off.
    if (reply === TRUNCATED_MARKER) return { text: "{\"benchmarks\": [{", truncated: true };
    return { text: reply, truncated: false };
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
      async generate(): Promise<GeneratedReply> {
        throw new Error("should never be called");
      },
    };
    await expect(dead.requireReady()).rejects.toBeInstanceOf(RuntimeUnavailableError);
  });
});

describe("answers about the prompt's example instead of the model", () => {
  const REASONING_CARD = "QwQ is the reasoning model of the Qwen series. It thinks before answering hard maths and coding problems.";
  const AUDIO_CARD = "This model takes an audio file and its transcript and returns word-level alignment for speech.";

  it("rejects the audio not_for that leaked out of the worked example", () => {
    // The real failure: QwQ-32B, a reasoning model, was given the example's
    // not_for verbatim. It reads perfectly and is about a different model.
    const leaked = {
      ...VALID,
      one_liner: "Takes a question, returns step-by-step reasoning and a final answer.",
      not_for: "transcribing audio, you need a separate speech recognition model for that",
      category: ["reasoning" as const],
    };
    const problems = proseChecks(leaked, REASONING_CARD);
    expect(problems.join(" ")).toContain("worked example");
  });

  it("allows the same sentence when the card really is about audio", () => {
    const grounded = {
      ...VALID,
      not_for: "transcribing audio, you need a separate speech recognition model for that",
    };
    expect(proseChecks(grounded, AUDIO_CARD)).toEqual([]);
  });

  it("says nothing when there is no card to check against", () => {
    expect(proseChecks(VALID)).toEqual([]);
  });

  it("retries a leaked answer and accepts the corrected one", async () => {
    const leaked = JSON.stringify({
      ...VALID,
      one_liner: "Takes a question, returns step-by-step reasoning and a final answer.",
      not_for: "transcribing audio, you need a separate speech recognition model for that",
    });
    const corrected = JSON.stringify({
      ...VALID,
      one_liner: "Takes a question, returns step-by-step reasoning and a final answer.",
      not_for: "quick replies, it works through problems at length before answering",
      category: ["reasoning"],
    });
    const runtime = new ScriptedRuntime([leaked, corrected]);
    const result = await extractProse(runtime, options, REASONING_CARD);
    expect(result.attempts).toHaveLength(2);
    expect(result.value.not_for).toContain("quick replies");
  });
});

describe("answers that echo the instructions back", () => {
  it("rejects the exact placeholder that reached a user", () => {
    // The real Qwen3.6 reply. It passed every other rule: right length, no
    // jargon, no ungrounded modality — because a sentence describing what a
    // sentence should contain is itself a well-formed sentence.
    const echoed = {
      ...VALID,
      one_liner: "What goes in, what comes out, what you would build with it. 25 words maximum.",
    };
    expect(proseChecks(echoed).join(" ")).toContain("repeats the instructions");
  });

  it.each([
    "two to four concrete things someone would use this for",
    "one sentence, or null",
    "…",
    "the most likely mistake, and what to use instead",
  ])("rejects the echoed value %s", (text) => {
    expect(looksLikeInstruction(text)).toBe(true);
  });

  it.each([
    "Takes a question and returns step-by-step reasoning and a final answer.",
    "subtitle timing",
    "transcribing audio, use a speech-to-text model instead",
  ])("accepts the real answer %s", (text) => {
    expect(looksLikeInstruction(text)).toBe(false);
  });

  it("checks every prose field, not only the one-liner", () => {
    const echoed = { ...VALID, use_for: ["two to four short phrases", "karaoke"] };
    expect(proseChecks(echoed).join(" ")).toContain("use_for[0]");
  });

  it("retries an echo and accepts the corrected answer", async () => {
    const echoed = JSON.stringify({
      ...VALID,
      one_liner: "What goes in, what comes out, what you would build with it. 25 words maximum.",
    });
    const runtime = new ScriptedRuntime([echoed, validJson]);
    const result = await extractProse(runtime, options);
    expect(result.attempts).toHaveLength(2);
    expect(result.value.one_liner).toBe(VALID.one_liner);
  });
});

describe("not_for names what the model cannot do", () => {
  const REASONING_CARD = "QwQ is the reasoning model of the Qwen series. It thinks before answering hard maths and coding problems.";

  it("allows a warning about a capability the README never mentions", () => {
    // This is the point of not_for. Rejecting it failed QwQ-32B outright,
    // three retries and no answer at all.
    const warned = {
      ...VALID,
      one_liner: "Takes a question, returns step-by-step reasoning and a final answer.",
      not_for: "generating images; use an image model for that",
      category: ["reasoning" as const],
    };
    expect(proseChecks(warned, REASONING_CARD)).toEqual([]);
  });

  it.each([
    "creating videos from a prompt; use a video model",
    "producing music; use a music model instead",
    "embedding documents for search; use an embedding model",
  ])("allows the warning %s on a text model", (not_for) => {
    const warned = { ...VALID, one_liner: "Takes a question, returns a reasoned answer.", not_for };
    expect(proseChecks(warned, REASONING_CARD)).toEqual([]);
  });

  it("still rejects the worked example's own sentence on an unrelated card", () => {
    const leaked = {
      ...VALID,
      one_liner: "Takes a question, returns step-by-step reasoning and a final answer.",
      not_for: "transcribing audio, you need a separate speech recognition model for that",
    };
    expect(proseChecks(leaked, REASONING_CARD).join(" ")).toContain("worked example");
  });

  it("still rejects a one_liner about a modality the card never mentions", () => {
    const wrong = { ...VALID, one_liner: "Takes an audio file and returns word timings." };
    expect(proseChecks(wrong, REASONING_CARD).join(" ")).toContain("one_liner talks about audio");
  });
});

describe("tidying hyphenated list items", () => {
  it("lowers a hyphenated opener", () => {
    expect(uncapitalise("Real-time speech analysis")).toBe("real-time speech analysis");
  });

  it("leaves a hyphenated model name alone", () => {
    expect(uncapitalise("Qwen3-ASR pairing")).toBe("Qwen3-ASR pairing");
  });
});

describe("a reply the runtime cut off", () => {
  it("is reported as cut off, not as malformed", async () => {
    // The real Qwen3.6 failure: a 62-row comparison table produced hundreds of
    // JSON entries and stopped mid-array. Calling that "the reply was not JSON"
    // points at the model's writing when the fault is the room it was given.
    const runtime = new ScriptedRuntime([TRUNCATED_MARKER]);
    await expect(extractProse(runtime, options)).rejects.toBeInstanceOf(TruncatedReplyError);
  });

  it("names the setting that would fix it", async () => {
    const runtime = new ScriptedRuntime([TRUNCATED_MARKER]);
    try {
      await extractProse(runtime, options);
      expect.unreachable("should have thrown");
    } catch (error) {
      expect((error as TruncatedReplyError).message).toContain("cut off");
      expect((error as TruncatedReplyError).fix).toContain("benchmarkTokenBudget");
    }
  });

  it("does not waste retries on a limit that will be hit again", async () => {
    const runtime = new ScriptedRuntime([TRUNCATED_MARKER, TRUNCATED_MARKER, TRUNCATED_MARKER]);
    await expect(extractProse(runtime, options)).rejects.toThrow();
    // Asking again with the same room gets the same cut-off reply.
    expect(runtime.requests).toHaveLength(1);
  });
});
